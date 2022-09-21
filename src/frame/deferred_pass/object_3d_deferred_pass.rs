// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use cgmath::{Matrix3, Rad, Matrix4, Point3, Vector3};
use vulkano::format::Format;
use vulkano::image::ImageDimensions;
use vulkano::image::ImmutableImage;
use vulkano::image::MipmapsCount;
use vulkano::image::view::ImageView;
use vulkano::render_pass::RenderPass;
use vulkano::sampler::Filter;
use vulkano::sampler::Sampler;
use vulkano::sampler::SamplerAddressMode;
use vulkano::sampler::SamplerCreateInfo;
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
    render_pass::Subpass,
};

use crate::scene_pkg::mesh::{ Normal, Tangent, Uv, Vertex };
use crate::scene_pkg::object3d::Object3D;
use crate::system::object_3d_draw_system::Object3DDrawSystem;

use super::deferred_map_renderer::DeferredMapRenderer;

#[repr(C)]
#[derive(Clone)]
pub struct Buffers {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub normals_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    pub uv_buffer: Arc<CpuAccessibleBuffer<[Uv]>>,
    pub tangent_buffer: Arc<CpuAccessibleBuffer<[Tangent]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    pub uniform_buffer: CpuBufferPool<vs::ty::Data>,
}

pub struct Object3DDeferredPass{
    gfx_queue: Arc<Queue>,
    object_3d: Object3D,
    pipeline: Arc<GraphicsPipeline>,
    subpass: Subpass, 
    buffers: Buffers,
    albedo_set: Arc<PersistentDescriptorSet>,
    normal_set: Arc<PersistentDescriptorSet>,
}

impl Object3DDeferredPass{
    /// Initializes a triangle drawing system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        object_3d: Object3D
    ) -> Object3DDeferredPass {

        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let buffers = Object3DDeferredPass::create_buffers(gfx_queue.clone(), object_3d.clone());

        let pipeline = Object3DDeferredPass::create_pipeline(gfx_queue.clone(), subpass.clone());

        let albedo_set = Object3DDeferredPass::create_albedo_set(gfx_queue.clone(), pipeline.clone(), object_3d.clone());
    
        let normal_set = Object3DDeferredPass::create_normal_set(gfx_queue.clone(), pipeline.clone(), object_3d.clone());

        Object3DDeferredPass {
            gfx_queue,
            object_3d,
            pipeline,
            subpass,
            buffers,
            albedo_set,
            normal_set
        }
    }


    pub fn draw(&mut self, viewport_dimensions: [u32; 2], world: Matrix4<f32>, projection: Matrix4<f32>, view: Matrix4<f32>) -> SecondaryAutoCommandBuffer {

        //descriptor set
        let uniform_buffer_subbuffer = {

            let uniform_data = vs::ty::Data {
                model: self.object_3d.model_matrix.into(),
                world: world.into(),
                view: (view).into(),
                proj: projection.into(),
            };

            self.buffers.uniform_buffer.next(uniform_data).unwrap()
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
                self.albedo_set.clone(),
            )
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                2,
                self.normal_set.clone(),
            )
            .bind_vertex_buffers(0, (self.buffers.vertex_buffer.clone(), self.buffers.normals_buffer.clone(), self.buffers.uv_buffer.clone(), self.buffers.tangent_buffer.clone()))
            .bind_index_buffer(self.buffers.index_buffer.clone())
            .draw_indexed(self.buffers.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    // private methods

    fn create_pipeline(gfx_queue: Arc<Queue>, subpass: Subpass) -> Arc<GraphicsPipeline> {
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
    }

    fn create_normal_set(gfx_queue: Arc<Queue>, pipeline: Arc<GraphicsPipeline>, object_3d: Object3D) -> Arc<PersistentDescriptorSet> {
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
                MipmapsCount::One,
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

        normal_set
    }

    fn create_albedo_set(gfx_queue: Arc<Queue>, pipeline: Arc<GraphicsPipeline>, object_3d: Object3D) -> Arc<PersistentDescriptorSet> {
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
        let diffuse_set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(0, texture, sampler)],
        )
        .unwrap();

        diffuse_set
    }

    pub fn create_buffers(gfx_queue: Arc<Queue>,  object_3d: Object3D) -> Buffers {
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

        Buffers{vertex_buffer, normals_buffer, uv_buffer, tangent_buffer, index_buffer, uniform_buffer}

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

layout(location = 0) out vec4 v_position;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out mat3 v_tbn;

layout(set = 0, binding = 0) uniform Data {
    mat4 model;
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view  * uniforms.model * uniforms.world;
    mat4 model_view = uniforms.view * uniforms.model;
    v_normal = mat3(worldview) * normal;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
    v_uv = uv;

    v_position = uniforms.model * vec4(position, 1.0);

    vec3 t = normalize(vec3(model_view * vec4(tangent,   0.0)));
    vec3 n = normalize(vec3(model_view * vec4(normal,    0.0)));

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

layout(location = 0) in vec4 v_position;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in mat3 v_tbn;

layout(location = 0) out vec4 f_position;
layout(location = 1) out vec4 f_color;
layout(location = 2) out vec3 f_normal;

layout(set = 1, binding = 0) uniform sampler2D tex;
layout(set = 2, binding = 0) uniform sampler2D normal_map;

void main() {

    f_position = v_position;

    f_color = texture(tex, v_uv);
    //f_color = vec4(0.5, 0.5, 0.5, 1.0);

    f_normal = texture(normal_map, v_uv).rgb;
    f_normal = (f_normal * 2.0 - 1.0);
    f_normal = normalize(v_tbn * f_normal);
}"
    }
}
