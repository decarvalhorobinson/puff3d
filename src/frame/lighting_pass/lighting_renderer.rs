use std::sync::{Arc, Mutex};
use vulkano::{format::Format, sync::GpuFuture};

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SecondaryCommandBuffer, SubpassContents,
    },
    device::Queue,
    image::ImageViewAbstract,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
};

use crate::scene_pkg::scene::Scene;

use super::lighting_pass::LightingPass;

pub struct LightingRenderer {
    pub scene: Arc<Mutex<Scene>>,
    pub gfx_queue: Arc<Queue>,
    pub render_pass: Arc<RenderPass>,
    pub lighting_passes: Vec<LightingPass>,

    pub framebuffer: Option<Arc<Framebuffer>>,
    pub command_buffer_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,
}

impl LightingRenderer {
    pub fn new(
        gfx_queue: Arc<Queue>,
        scene: Arc<Mutex<Scene>>,
        final_output_format: Format,
    ) -> LightingRenderer {
        let render_pass = vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                // Will be bound to `self.position_buffer`.
                final_color: {
                    load: Clear,
                    store: Store,
                    format: final_output_format,
                    samples: 1,
                }
            },
            passes: [
                {
                    color: [final_color],
                    depth_stencil: {},
                    input: []
                }
            ]
        )
        .unwrap();

        let scene_locked = scene.lock().unwrap();
        let mut lighting_passes: Vec<LightingPass> = vec![];
        lighting_passes.reserve(scene_locked.objects.len());
        for dir_light in scene_locked.directional_lights.clone() {
            lighting_passes.push(LightingPass::new(
                gfx_queue.clone(),
                render_pass.clone(),
                dir_light,
            ));
        }

        LightingRenderer {
            scene: scene.clone(),
            gfx_queue: gfx_queue.clone(),
            render_pass: render_pass,
            lighting_passes: lighting_passes,
            framebuffer: Option::None,
            command_buffer_builder: Option::None,
        }
    }

    pub fn draw(
        &mut self,
        shadow_image: Arc<dyn ImageViewAbstract + 'static>,
        position_image: Arc<dyn ImageViewAbstract + 'static>,
        color_image: Arc<dyn ImageViewAbstract + 'static>,
        normals_image: Arc<dyn ImageViewAbstract + 'static>,
    ) {
        let view;
        let world;
        {
            let scene_locked = self.scene.lock().unwrap();
            world = scene_locked.world_model;
            view = scene_locked.active_camera.get_view_matrix();
        }

        for i in 0..self.lighting_passes.len() {
            let cb = self.lighting_passes[i].draw(
                self.framebuffer.clone().unwrap().extent(),
                world,
                view,
                shadow_image.clone(),
                position_image.clone(),
                color_image.clone(),
                normals_image.clone(),
            );
            self.execute_draw_pass(cb);
        }
    }

    pub fn begin_render_pass(&mut self, final_image: Arc<dyn ImageViewAbstract + 'static>) {
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![final_image.clone()],
                ..Default::default()
            },
        )
        .unwrap();

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some(1.0f32.into()),
                    ],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap();
        self.framebuffer = Some(framebuffer);
        self.command_buffer_builder = Some(command_buffer_builder);
    }

    pub fn execute_draw_pass<C>(&mut self, command_buffer: C)
    where
        C: SecondaryCommandBuffer + 'static,
    {
        self.command_buffer_builder
            .as_mut()
            .unwrap()
            .execute_commands(command_buffer)
            .unwrap();
    }

    pub fn end_render_pass<F: GpuFuture + 'static>(
        &mut self,
        future: F,
    ) -> vulkano::command_buffer::CommandBufferExecFuture<F, PrimaryAutoCommandBuffer> {
        self.command_buffer_builder
            .as_mut()
            .unwrap()
            .end_render_pass()
            .unwrap();
        let command_buffer = self.command_buffer_builder.take().unwrap().build().unwrap();

        future
            .then_execute(self.gfx_queue.clone(), command_buffer)
            .unwrap()
    }
}
