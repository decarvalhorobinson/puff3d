use std::sync::{Arc, Mutex};
use vulkano::{
    format::Format,
    image::{AttachmentImage, ImageUsage},
    sync::GpuFuture,
};

use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        RenderPassBeginInfo, SecondaryCommandBuffer, SubpassContents,
    },
    device::Queue,
    image::view::ImageView,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
};

use crate::scene_pkg::scene::Scene;

use super::volume_pass::VolumePass;

pub struct VolumeMapRenderer {
    pub scene: Arc<Mutex<Scene>>,
    pub gfx_queue: Arc<Queue>,
    pub render_pass: Arc<RenderPass>,
    pub volume_passes: Vec<VolumePass>,

    pub framebuffer: Option<Arc<Framebuffer>>,
    pub command_buffer_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,

    pub final_image: Arc<ImageView<AttachmentImage>>,
    pub depth_image: Arc<ImageView<AttachmentImage>>,
    
}

impl VolumeMapRenderer {
    pub fn new(gfx_queue: Arc<Queue>, scene: Arc<Mutex<Scene>>) -> VolumeMapRenderer {
        let render_pass = vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                final_color: {
                    load: Clear,
                    store: Store,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            passes: [
                {
                    color: [final_color],
                    depth_stencil: {depth},
                    input: []
                }
            ]
        )
        .unwrap();

        let final_image = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [2048, 2048],
                Format::R16G16B16A16_SFLOAT,
                ImageUsage {
                    transient_attachment: false,
                    input_attachment: false,
                    sampled: true,
                    ..ImageUsage::empty()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let depth_image = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [2048, 2048],
                Format::D16_UNORM,
                ImageUsage {
                    transient_attachment: true,
                    input_attachment: true,
                    ..ImageUsage::empty()
                },
            )
            .unwrap(),
        )
        .unwrap();

        let scene_locked = scene.lock().unwrap();
        let mut volume_passes: Vec<VolumePass> = vec![];
        volume_passes.reserve(scene_locked.objects.len());
        for volume in scene_locked.volumes.clone() {
            volume_passes.push(VolumePass::new(
                gfx_queue.clone(),
                render_pass.clone(),
                volume,
            ));
        }

        VolumeMapRenderer {
            scene: scene.clone(),
            gfx_queue: gfx_queue.clone(),
            render_pass: render_pass,
            volume_passes: volume_passes,
            framebuffer: Option::None,
            command_buffer_builder: Option::None,

            final_image: final_image,
            depth_image: depth_image,
        }
    }

    pub fn draw(&mut self, delta_time: u128) {
        let view;
        let world;
        let eye_pos;
        {
            let scene_locked = self.scene.lock().unwrap();
            world = scene_locked.world_model;
            view = scene_locked.active_camera.get_view_matrix();
            eye_pos = scene_locked.active_camera.position;
        }

        for i in 0..self.volume_passes.len() {
            let cb = self.volume_passes[i].draw(
                self.framebuffer.clone().unwrap().extent(),
                world, 
                view, 
                eye_pos,
                delta_time
            );
            self.execute_draw_pass(cb);
        }
    }

    pub fn begin_render_pass(&mut self) {
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![
                    self.final_image.clone(),
                    self.depth_image.clone(),
                ],
                ..Default::default()
            },
        )
        .unwrap();
        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            self.gfx_queue.device().clone(),
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        command_buffer_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
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

    #[inline]
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        self.framebuffer.clone().unwrap().extent()
    }
}
