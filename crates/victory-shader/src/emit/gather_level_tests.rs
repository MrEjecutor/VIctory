//! GPU checks for gathers that use their explicit mip level.

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::HostShader;
use super::host::Host;
use super::{lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const WIDTH: u32 = 8;
const MIP_LEVELS: u16 = 3;

#[derive(Clone, Copy, Debug)]
struct Case {
    level: u32,
    u: f32,
    v: f32,
    offset: Option<(i32, i32)>,
    expected: [f32; 4],
}

fn table(user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    }
}

fn used(
    kind: ResourceType,
    table_offset_dwords: u32,
    dwords: u8,
    user_data: u8,
    slot: u8,
) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    }
}

fn v_mov_literal(words: &mut Vec<u32>, reg: u8, value: u32) {
    words.extend([
        0x7E00_0000 | (u32::from(reg) << 17) | (1 << 9) | 0xFF,
        value,
    ]);
}

fn buffer_store_dword(reg: u8, byte_offset: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | u32::from(byte_offset),
        0x80 << 24 | (10 << 16) | (u32::from(reg) << 8),
    ]
}

/// Builds a compute shader with constant input coordinates and stores each
/// gather's four channels to consecutive dwords of a raw output buffer.
fn build(cases: &[Case]) -> HostShader {
    let mut words = vec![
        0xBE82_0301, // s_mov_b32 s2, s1
        0xBE81_0380, // s_mov_b32 s1, 0
        0xBE83_0380, // s_mov_b32 s3, 0
        // The output buffer's V# first: loading the T# into s[0:7] below
        // overwrites s2.
        0xBE84_0302, // s_mov_b32 s4, s2
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0A02,
        0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
        0xF408_0200,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], null
        0xF40C_0001,
        0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
    ];
    for (n, case) in cases.iter().enumerate() {
        if let Some((x, y)) = case.offset {
            let offset = (x as u32 & 0x3F) | ((y as u32 & 0x3F) << 8);
            v_mov_literal(&mut words, 2, offset);
            v_mov_literal(&mut words, 3, case.u.to_bits());
            v_mov_literal(&mut words, 4, case.v.to_bits());
            v_mov_literal(&mut words, 5, (case.level as f32).to_bits());
            words.extend([0xF150_0108, 0x0040_0002]); // image_gather4_l_o
        } else {
            v_mov_literal(&mut words, 2, case.u.to_bits());
            v_mov_literal(&mut words, 3, case.v.to_bits());
            v_mov_literal(&mut words, 4, (case.level as f32).to_bits());
            words.extend([0xF110_0108, 0x0040_0002]); // image_gather4_l
        }
        for channel in 0..4 {
            words.extend(buffer_store_dword(
                channel,
                ((n * 4 + usize::from(channel)) * 4) as u16,
            ));
        }
    }
    words.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![table(0, 0), table(1, 1)],
        used: vec![
            used(ResourceType::Sampler, 0, 4, 0, 0),
            used(ResourceType::Srv, 0, 8, 1, 1),
            used(ResourceType::Uav, 16, 4, 1, 1),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    }
}

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

fn make_resource(
    device: &ID3D12Device,
    heap_type: D3D12_HEAP_TYPE,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Result<ID3D12Resource, String> {
    let mut resource = None;
    // SAFETY: a plain D3D12 call with descriptions that outlive it.
    unsafe {
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap_type,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                desc,
                state,
                None,
                &mut resource,
            )
            .map_err(err)?;
    }
    resource.ok_or_else(|| "no resource".to_string())
}

fn buffer_desc(bytes: u64, flags: D3D12_RESOURCE_FLAGS) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    }
}

fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: borrows the interface pointer without a reference;
                // `resource` outlives the barrier's use.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// Uploads an 8x8 RGBA texture with three distinct mip levels, dispatches
/// the shader, and reads its raw UAV back.
fn dispatch(host: &Host, shader: &HostShader, output_dwords: u32) -> Result<Vec<u32>, String> {
    let device = &host.device;
    // SAFETY: D3D12 calls with descriptions that outlive them; every resource
    // stays alive until the fence says the GPU is done.
    unsafe {
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(err)?;
        let allocator: ID3D12CommandAllocator = device
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(err)?;
        let (root, pso) = host.compute_pipeline(shader)?;
        let list: ID3D12GraphicsCommandList = device
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pso)
            .map_err(err)?;

        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(WIDTH),
            Height: WIDTH,
            DepthOrArraySize: 1,
            MipLevels: MIP_LEVELS,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        let texture = make_resource(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            &texture_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let upload_desc = buffer_desc(3584, D3D12_RESOURCE_FLAG_NONE);
        let upload = make_resource(
            device,
            D3D12_HEAP_TYPE_UPLOAD,
            &upload_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut mapped = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut mapped)).map_err(err)?;
        for level in 0..u32::from(MIP_LEVELS) {
            let width = (WIDTH >> level).max(1);
            let height = (WIDTH >> level).max(1);
            let offset = match level {
                0 => 0usize,
                1 => 2048,
                _ => 3072,
            };
            for y in 0..height {
                let row = mapped.cast::<u8>().add(offset + y as usize * 256);
                for x in 0..width {
                    let texel = [(level * 1000 + y * 100 + x) as f32, 0.0, 0.0, 1.0];
                    row.add(x as usize * 16)
                        .cast::<f32>()
                        .copy_from_nonoverlapping(texel.as_ptr(), 4);
                }
            }
        }
        upload.Unmap(0, None);

        for level in 0..u32::from(MIP_LEVELS) {
            let width = (WIDTH >> level).max(1);
            let height = (WIDTH >> level).max(1);
            let offset = match level {
                0 => 0u64,
                1 => 2048,
                _ => 3072,
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&upload),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: offset,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: width,
                            Height: height,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: level,
                },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        list.ResourceBarrier(&[transition(
            &texture,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        )]);

        let output_bytes = u64::from(output_dwords) * 4;
        let output_desc = buffer_desc(output_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        let output = make_resource(
            device,
            D3D12_HEAP_TYPE_DEFAULT,
            &output_desc,
            // Buffers start in COMMON and are promoted to UAV on first use.
            D3D12_RESOURCE_STATE_COMMON,
        )?;
        let readback_desc = buffer_desc(output_bytes, D3D12_RESOURCE_FLAG_NONE);
        let readback = make_resource(
            device,
            D3D12_HEAP_TYPE_READBACK,
            &readback_desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        let view_heap: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .map_err(err)?;
        let view_size = device
            .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
            as usize;
        let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MostDetailedMip: 0,
                    MipLevels: u32::from(MIP_LEVELS),
                    ..Default::default()
                },
            },
        };
        let mut at = view_heap.GetCPUDescriptorHandleForHeapStart();
        device.CreateShaderResourceView(&texture, Some(&srv), at);
        at.ptr += view_size;
        device.CreateShaderResourceView(&texture, Some(&srv), at);
        at.ptr += view_size;
        let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: output_dwords,
                    Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                    ..Default::default()
                },
            },
        };
        device.CreateUnorderedAccessView(&output, None, Some(&uav), at);

        let sampler_heap: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .map_err(err)?;
        device.CreateSampler(
            &D3D12_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                MaxAnisotropy: 1,
                MaxLOD: f32::MAX,
                ..Default::default()
            },
            sampler_heap.GetCPUDescriptorHandleForHeapStart(),
        );

        let mut user_data = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for item in &shader.user_data {
            assert_eq!(item.root, RootType::Table);
            user_data[item.index as usize] = 0;
        }
        // Clones: the list does not keep the heaps alive until it runs.
        list.SetDescriptorHeaps(&[Some(view_heap.clone()), Some(sampler_heap.clone())]);
        list.SetComputeRootSignature(&root);
        list.SetComputeRoot32BitConstants(0, user_data.len() as u32, user_data.as_ptr().cast(), 0);
        list.Dispatch(1, 1, 1);
        list.ResourceBarrier(&[transition(
            &output,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.CopyBufferRegion(&readback, 0, &output, 0, output_bytes);
        list.Close().map_err(err)?;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        queue.Signal(&fence, 1).map_err(err)?;
        while fence.GetCompletedValue() < 1 {
            if let Err(e) = device.GetDeviceRemovedReason() {
                return Err(err(e));
            }
            std::thread::yield_now();
        }
        device.GetDeviceRemovedReason().map_err(err)?;
        let mut mapped = std::ptr::null_mut();
        readback.Map(0, None, Some(&mut mapped)).map_err(err)?;
        let out = std::slice::from_raw_parts(mapped.cast::<u32>(), output_dwords as usize).to_vec();
        readback.Unmap(0, None);
        Ok(out)
    }
}

#[test]
fn explicit_level_gathers_read_the_selected_mips_and_offsets() {
    let cases = [
        Case {
            level: 1,
            u: 0.5,
            v: 0.5,
            offset: None,
            expected: [1201.0, 1202.0, 1102.0, 1101.0],
        },
        Case {
            level: 1,
            u: 0.125,
            v: 0.75,
            offset: None,
            expected: [1300.0, 1301.0, 1201.0, 1200.0],
        },
        Case {
            level: 2,
            u: 0.875,
            v: 0.375,
            offset: None,
            expected: [2101.0, 2101.0, 2001.0, 2001.0],
        },
        Case {
            level: 1,
            u: 0.5,
            v: 0.5,
            offset: Some((1, -1)),
            expected: [1102.0, 1103.0, 1003.0, 1002.0],
        },
        Case {
            level: 2,
            u: 0.5,
            v: 0.5,
            offset: Some((-1, 1)),
            expected: [2100.0, 2100.0, 2100.0, 2100.0],
        },
        Case {
            level: 2,
            u: 0.875,
            v: 0.125,
            offset: Some((1, 1)),
            expected: [2101.0, 2101.0, 2001.0, 2001.0],
        },
    ];
    let shader = build(&cases);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let got = dispatch(&host, &shader, (cases.len() * 4) as u32)
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        ran = true;
        for (n, case) in cases.iter().enumerate() {
            let words = &got[n * 4..n * 4 + 4];
            let got = [
                f32::from_bits(words[0]),
                f32::from_bits(words[1]),
                f32::from_bits(words[2]),
                f32::from_bits(words[3]),
            ];
            assert_eq!(got, case.expected, "{label}: gather case {n}: {case:?}");
        }
    }
    assert!(ran, "neither hardware nor WARP SM 6.6 device available");
}
