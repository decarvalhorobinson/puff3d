// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::{Matrix3, Rad, Matrix4, Point3, Vector3};
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use std::{sync::Arc, io::Cursor};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, TypedBufferAccess, CpuBufferPool},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
        SecondaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Queue,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::Subpass, image::{ImageDimensions, ImmutableImage, MipmapsCount, view::ImageView}, format::Format, sampler::{Sampler, SamplerCreateInfo, SamplerAddressMode, Filter},
};

use crate::scene_pkg::mesh::{ Normal, Tangent, Uv, Vertex };
use crate::scene_pkg::object3d::Object3D;

#[repr(C)]
#[derive(Clone)]
pub struct Object3DDrawSystem {
    gfx_queue: Arc<Queue>,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    subpass: Subpass,
    pipeline: Arc<GraphicsPipeline>,
    normals_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    uniform_buffer: CpuBufferPool<vs::ty::Data>,
    rotation_start: f32,
    uv_buffer: Arc<CpuAccessibleBuffer<[Uv]>>,
    texture_set: Arc<PersistentDescriptorSet>,
    tangent_buffer: Arc<CpuAccessibleBuffer<[Tangent]>>,
    normal_set: Arc<PersistentDescriptorSet>,
    object_3d: Object3D
}

impl Object3DDrawSystem {
    /// Initializes a triangle drawing system.
    pub fn new(
        gfx_queue: Arc<Queue>, 
        subpass: Subpass, 
        object_3d: Object3D
    ) -> Object3DDrawSystem {
        let rotation_start = 0.0;
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::none()
                },
                false,
                object_3d.mesh.vertices.clone(),
            )
            .expect("failed to create buffer")
        };

        let normals_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::none()
                },
                false,
                object_3d.mesh.normals.clone(),
            )
            .expect("failed to create buffer")
        };

        let uv_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::none()
                },
                false,
                object_3d.mesh.uvs.clone(),
            )
            .expect("failed to create buffer")
        };

        let tangent_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::none()
                },
                false,
                object_3d.mesh.tangent.clone(),
            )
            .expect("failed to create buffer")
        };

        let index_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    index_buffer: true,
                    ..BufferUsage::none()
                },
                false,
                object_3d.mesh.indices.clone(),
            )
            .expect("failed to create buffer")
        };

        let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(
            gfx_queue.device().clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::none()
            },
        );

        let pipeline = {
            let vs = vs::load(gfx_queue.device().clone()).expect("failed to create shader module");
            let fs = fs::load(gfx_queue.device().clone()).expect("failed to create shader module");

            GraphicsPipeline::start()
                .vertex_input_state(
                    BuffersDefinition::new()
                        .vertex::<Vertex>()
                        .vertex::<Normal>()
                        .vertex::<Uv>()
                        .vertex::<Tangent>(),
                )
                .vertex_shader(vs.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .fragment_shader(fs.entry_point("main").unwrap(), ())
                .depth_stencil_state(DepthStencilState::simple_depth_test())
                .render_pass(subpass.clone())
                .build(gfx_queue.device().clone())
                .unwrap()
        };


        //texture image
        let (texture, tex_future) = {
            let f = File::open(object_3d.material.diffuse_file_path.clone()).unwrap();
            let mut reader = BufReader::new(f);
            let mut png_bytes = Vec::new();
            reader.read_to_end(&mut png_bytes).unwrap();

            let cursor = Cursor::new(png_bytes);
            let decoder = png::Decoder::new(cursor);
            let mut reader = decoder.read_info().unwrap();
            let info = reader.info();
            let dimensions = ImageDimensions::Dim2d {
                width: info.width,
                height: info.height,
                array_layers: 1,
            };
            let mut image_data = Vec::new();
            image_data.resize((info.width * info.height * 6) as usize, 0);
            reader.next_frame(&mut image_data).unwrap();
    
            let (image, future) = ImmutableImage::from_iter(
                image_data,
                dimensions,
                MipmapsCount::Log2,
                Format::R8G8B8A8_SRGB,
                gfx_queue.clone(),
            )
            .unwrap();
            (ImageView::new_default(image).unwrap(), future)
        };
    
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let layout = pipeline.layout().set_layouts().get(1).unwrap();
        let texture_set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, texture, sampler)],
        )
        .unwrap();

        // normal
        //texture image
        let (normal, normal_future) = {
            let f = File::open(object_3d.material.normal_file_path.clone()).unwrap();
            let mut reader = BufReader::new(f);
            let mut png_bytes = Vec::new();
            reader.read_to_end(&mut png_bytes).unwrap();

            let cursor = Cursor::new(png_bytes);
            let decoder = png::Decoder::new(cursor);
            let mut reader = decoder.read_info().unwrap();
            let info = reader.info();
            let dimensions = ImageDimensions::Dim2d {
                width: info.width,
                height: info.height,
                array_layers: 1,
            };
            let mut image_data = Vec::new();
            image_data.resize((info.width * info.height * 4) as usize, 0);
            reader.next_frame(&mut image_data).unwrap();
    
            let (image, future) = ImmutableImage::from_iter(
                image_data,
                dimensions,
                MipmapsCount::Log2,
                Format::R8G8B8A8_UNORM,
                gfx_queue.clone(),
            )
            .unwrap();
            (ImageView::new_default(image).unwrap(), future)
        };
    
        let normal_sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        let layout = pipeline.layout().set_layouts().get(2).unwrap();
        let normal_set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, normal, normal_sampler)],
        )
        .unwrap();

        Object3DDrawSystem {
            object_3d,
            rotation_start,
            gfx_queue,
            vertex_buffer,
            normals_buffer,
            uv_buffer,
            tangent_buffer,
            index_buffer,
            uniform_buffer,
            subpass,
            pipeline,
            texture_set,
            normal_set
        }
    }

    /// Builds a secondary command buffer that draws the triangle on the current subpass.
    pub fn draw(&mut self, viewport_dimensions: [u32; 2], screen_to_world: Matrix4<f32>, start_rot: f32, scale: f32) -> SecondaryAutoCommandBuffer {


        //descriptor set
        let uniform_buffer_subbuffer = {
            
            self.rotation_start = self.rotation_start + 0.01;

            let angle = self.rotation_start + start_rot;
            let rotation = Matrix3::from_axis_angle(Vector3::new(0.0, 0.0, 1.0), Rad(0 as f32));
            
            let rotation = rotation * Matrix3::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(angle as f32));
            let rotation = Matrix4::from(rotation) * screen_to_world;
            let view = Matrix4::look_at_rh(
                Point3::new(0.6, 1.0, 1.5),
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, -1.0, 0.0),
            );
            


            // note: this teapot was meant for OpenGL where the origin is at the lower left
            //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
            let aspect_ratio = viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32;
            let proj = cgmath::perspective(
                Rad(std::f32::consts::FRAC_PI_4),
                aspect_ratio,
                0.01,
                100.0,
            );
            
            let scale = Matrix4::from_scale(scale);

            let uniform_data = vs::ty::Data {
                world: rotation.into(),
                view: (view * scale).into(),
                proj: proj.into(),
            };

            self.uniform_buffer.next(uniform_data).unwrap()
        };

        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
        )
        .unwrap();

        //end descriptor set

        let mut builder = AutoCommandBufferBuilder::secondary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::MultipleSubmit,
            CommandBufferInheritanceInfo {
                render_pass: Some(self.subpass.clone().into()),
                ..Default::default()
            },
        )
        .unwrap();
        builder
            .set_viewport(
                0,
                [Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [viewport_dimensions[0] as f32, viewport_dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                1,
                self.texture_set.clone(),
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                2,
                self.normal_set.clone(),
            )
            .bind_vertex_buffers(0, (self.vertex_buffer.clone(), self.normals_buffer.clone(), self.uv_buffer.clone(), self.tangent_buffer.clone()))
            .bind_index_buffer(self.index_buffer.clone())
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 tangent;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out mat3 v_tbn;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    v_normal = mat3(worldview) * normal;
    //v_normal = normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    v_uv = uv;

    vec3 t = normalize(vec3(worldview * vec4(tangent,   0.0)));
    vec3 n = normalize(vec3(worldview * vec4(normal,    0.0)));

    t = normalize(t- dot(t, n) * n);
    vec3 b = cross(n, t);

    v_tbn = mat3(t, b, n);
}",
types_meta: {
    use bytemuck::{Pod, Zeroable};

    #[derive(Clone, Copy, Zeroable, Pod)]
}
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in mat3 v_tbn;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;

layout(set = 1, binding = 0) uniform sampler2D tex;
layout(set = 2, binding = 0) uniform sampler2D normal_map;

void main() {
    f_color = texture(tex, v_uv);
    //f_color = vec4(0.5, 0.5, 0.5, 1.0);

    f_normal = texture(normal_map, v_uv).rgb;
    f_normal = -(f_normal * 2.0 - 1.0);
    f_normal = normalize(v_tbn * f_normal);
}"
    }
}
