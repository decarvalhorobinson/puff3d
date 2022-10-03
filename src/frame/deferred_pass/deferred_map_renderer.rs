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

use super::object_3d_deferred_pass::Object3DDeferredPass;

pub struct DeferredMapRenderer {
    pub scene: Arc<Mutex<Scene>>,
    pub gfx_queue: Arc<Queue>,
    pub render_pass: Arc<RenderPass>,
    pub object_3d_passes: Vec<Object3DDeferredPass>,

    pub framebuffer: Option<Arc<Framebuffer>>,
    pub command_buffer_builder: Option<AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>>,

    pub position_image: Arc<ImageView<AttachmentImage>>,
    pub albedo_specular_image: Arc<ImageView<AttachmentImage>>,
    pub normals_image: Arc<ImageView<AttachmentImage>>,
    pub mettalic_image: Arc<ImageView<AttachmentImage>>,
    pub roughness_image: Arc<ImageView<AttachmentImage>>,
    pub ao_image: Arc<ImageView<AttachmentImage>>,
    depth_image: Arc<ImageView<AttachmentImage>>,
    
}

impl DeferredMapRenderer {
    pub fn new(gfx_queue: Arc<Queue>, scene: Arc<Mutex<Scene>>) -> DeferredMapRenderer {
        let render_pass = vulkano::ordered_passes_renderpass!(gfx_queue.device().clone(),
            attachments: {
                // Will be bound to `self.position_image`.
                position: {
                    load: Clear,
                    store: Store,
                    format: Format::R16G16B16A16_SFLOAT,
                    samples: 1,
                },
                // Will be bound to `self.albedo_specular_image`.
                albedo_specular: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                // Will be bound to `self.normals_image`.
                normals: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                // Will be bound to `self.metallic_image`.
                mettalic: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                // Will be bound to `self.roughness_image`.
                roughness: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                // Will be bound to `self.ao_image`.
                ao: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                // Will be bound to `self.depth_image`.
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16_UNORM,
                    samples: 1,
                }
            },
            passes: [
                // Write to the diffuse, normals and depth attachments.
                {
                    color: [position, albedo_specular, normals, mettalic, roughness, ao],
                    depth_stencil: {depth},
                    input: []
                }
            ]
        )
        .unwrap();

        let position_image = ImageView::new_default(
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

        let base_color_image = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [2048, 2048],
                Format::R8G8B8A8_UNORM,
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
        let normals_image = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [2048, 2048],
                Format::R8G8B8A8_UNORM,
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
        let mettalic_image = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [2048, 2048],
                Format::R8G8B8A8_UNORM,
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
        let roughness_image = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [2048, 2048],
                Format::R8G8B8A8_UNORM,
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
        let ao_image = ImageView::new_default(
            AttachmentImage::with_usage(
                gfx_queue.device().clone(),
                [2048, 2048],
                Format::R8G8B8A8_UNORM,
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
        let mut object_3d_passes: Vec<Object3DDeferredPass> = vec![];
        object_3d_passes.reserve(scene_locked.objects.len());
        for object_3d in scene_locked.objects.clone() {
            object_3d_passes.push(Object3DDeferredPass::new(
                gfx_queue.clone(),
                render_pass.clone(),
                object_3d,
            ));
        }

        DeferredMapRenderer {
            scene: scene.clone(),
            gfx_queue: gfx_queue.clone(),
            render_pass: render_pass,
            object_3d_passes: object_3d_passes,
            framebuffer: Option::None,
            command_buffer_builder: Option::None,

            position_image: position_image,
            albedo_specular_image: base_color_image,
            normals_image: normals_image,
            mettalic_image,
            roughness_image,
            ao_image,
            depth_image: depth_image,
        }
    }

    pub fn draw(&mut self) {
        let projection = Scene::projection(self.viewport_dimensions());
        let view;
        let world;
        {
            let scene_locked = self.scene.lock().unwrap();
            world = scene_locked.world_model;
            view = scene_locked.active_camera.get_view_matrix();
        }

        for i in 0..self.object_3d_passes.len() {
            let cb = self.object_3d_passes[i].draw(
                self.framebuffer.clone().unwrap().extent(),
                world,
                projection,
                view,
            );
            self.execute_draw_pass(cb);
        }
    }

    pub fn begin_render_pass(&mut self) {
        let framebuffer = Framebuffer::new(
            self.render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![
                    self.position_image.clone(),
                    self.albedo_specular_image.clone(),
                    self.normals_image.clone(),
                    self.mettalic_image.clone(),
                    self.roughness_image.clone(),
                    self.ao_image.clone(),
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
                        Some([0.0, 0.0, 0.0, 0.0].into()),
                        Some([0.0, 0.0, 0.0, 0.0].into()),
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

    #[inline]
    pub fn viewport_dimensions(&self) -> [u32; 2] {
        self.framebuffer.clone().unwrap().extent()
    }
}
