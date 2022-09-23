use cgmath::Matrix4;
use std::sync::Arc;
use vulkano::pipeline::graphics::rasterization::CullMode;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::render_pass::RenderPass;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
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

use crate::scene_pkg::mesh::Vertex;
use crate::scene_pkg::object3d::Object3D;

pub struct Object3DShadowPass {
    gfx_queue: Arc<Queue>,
    object_3d: Object3D,
    pipeline_depth: Arc<GraphicsPipeline>,
    subpass: Subpass,
    vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    uniform_data_buffer: CpuBufferPool<vs_depth::ty::Data>,
}

impl Object3DShadowPass {
    /// Initializes a triangle drawing system.
    pub fn new(
        gfx_queue: Arc<Queue>,
        render_pass: Arc<RenderPass>,
        object_3d: Object3D,
    ) -> Object3DShadowPass {
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

        let (vertex_buffer, index_buffer, uniform_data_buffer) =
            Object3DShadowPass::create_buffers(gfx_queue.clone(), object_3d.clone());

        let pipeline_depth =
            Object3DShadowPass::create_depth_pipeline(gfx_queue.clone(), subpass.clone());

        Object3DShadowPass {
            gfx_queue,
            object_3d,
            pipeline_depth,
            subpass,
            vertex_buffer,
            index_buffer,
            uniform_data_buffer,
        }
    }

    pub fn draw(
        &mut self,
        viewport_dimensions: [u32; 2],
        world: Matrix4<f32>,
        projection: Matrix4<f32>,
        view: Matrix4<f32>,
    ) -> SecondaryAutoCommandBuffer {
        //descriptor set
        let uniform_buffer_subbuffer = {
            let scale = Matrix4::from_scale(1.0);

            let uniform_data = vs_depth::ty::Data {
                model: self.object_3d.model_matrix.into(),
                world: world.into(),
                view: (view * scale).into(),
                proj: projection.into(),
            };

            self.uniform_data_buffer.from_data(uniform_data).unwrap()
        };

        let layout = self.pipeline_depth.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
        )
        .unwrap();

        //end descriptor set

        let mut builder = AutoCommandBufferBuilder::secondary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.queue_family_index(),
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
            .bind_pipeline_graphics(self.pipeline_depth.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline_depth.layout().clone(),
                0,
                set,
            )
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .bind_index_buffer(self.index_buffer.clone())
            .draw_indexed(self.index_buffer.len() as u32, 1, 0, 0, 0)
            .unwrap();
        builder.build().unwrap()
    }

    // private methods

    fn create_depth_pipeline(gfx_queue: Arc<Queue>, subpass: Subpass) -> Arc<GraphicsPipeline> {
        let vs =
            vs_depth::load(gfx_queue.device().clone()).expect("failed to create shader module");
        let fs =
            fs_depth::load(gfx_queue.device().clone()).expect("failed to create shader module");

        GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::default().cull_mode(CullMode::Front))
            .render_pass(subpass.clone())
            .build(gfx_queue.device().clone())
            .unwrap()
    }

    fn create_buffers(
        gfx_queue: Arc<Queue>,
        object_3d: Object3D,
    ) -> (
        Arc<CpuAccessibleBuffer<[Vertex]>>,
        Arc<CpuAccessibleBuffer<[u16]>>,
        CpuBufferPool<vs_depth::ty::Data>,
    ) {
        let vertex_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    vertex_buffer: true,
                    ..BufferUsage::empty()
                },
                false,
                object_3d.mesh.vertices.clone(),
            )
            .expect("failed to create buffer")
        };

        let index_buffer = {
            CpuAccessibleBuffer::from_iter(
                gfx_queue.device().clone(),
                BufferUsage {
                    index_buffer: true,
                    ..BufferUsage::empty()
                },
                false,
                object_3d.mesh.indices.clone(),
            )
            .expect("failed to create buffer")
        };

        let uniform_buffer_depth = CpuBufferPool::<vs_depth::ty::Data>::new(
            gfx_queue.device().clone(),
            BufferUsage {
                uniform_buffer: true,
                ..BufferUsage::empty()
            },
        );

        (vertex_buffer, index_buffer, uniform_buffer_depth)
    }
}

mod vs_depth {
    vulkano_shaders::shader! {
            ty: "vertex",
            src: "
#version 450

layout(location = 0) in vec3 position;

layout(set = 0, binding = 0) uniform Data {
    mat4 model;
    mat4 world;
    mat4 view;
    mat4 proj;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.world * uniforms.model;
    gl_Position = uniforms.proj * worldview * vec4(position, 1.0);

}",
    types_meta: {
        use bytemuck::{Pod, Zeroable};

        #[derive(Clone, Copy, Zeroable, Pod)]
    }
        }
}

mod fs_depth {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450


void main() {
}"
    }
}
