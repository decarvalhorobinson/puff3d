use std::sync::{Arc, Mutex};

use cgmath::{Matrix4, Point3, Rad, Vector3};
use vulkano::{
    command_buffer::PrimaryAutoCommandBuffer, device::Queue, format::Format,
    image::ImageViewAbstract, sync::GpuFuture,
};
use winit::event::VirtualKeyCode;

use crate::{scene_pkg::{camera::CameraMovement, scene::Scene}, config::input::Input};

use super::{
    deferred_pass::deferred_map_renderer::DeferredMapRenderer,
    lighting_pass::lighting_renderer::LightingRenderer,
    shadow_pass::shadow_map_renderer::ShadowMapRenderer,
};

pub struct SceneRenderer {
    scene: Arc<Mutex<Scene>>,
    shadow_map_renderer: ShadowMapRenderer,
    deferred_map_renderer: DeferredMapRenderer,
    lighting_renderer: LightingRenderer,

    rotation_light: f32,
}

impl SceneRenderer {
    pub fn new(
        gfx_queue: Arc<Queue>,
        scene: Arc<Mutex<Scene>>,
        swapchain_image_format: Format,
    ) -> SceneRenderer {
        let shadow_map_renderer = ShadowMapRenderer::new(gfx_queue.clone(), scene.clone());
        let deferred_map_renderer = DeferredMapRenderer::new(gfx_queue.clone(), scene.clone());
        let lighting_renderer =
            LightingRenderer::new(gfx_queue.clone(), scene.clone(), swapchain_image_format);

        SceneRenderer {
            scene,
            shadow_map_renderer,
            deferred_map_renderer,
            lighting_renderer,
            rotation_light: 0.0,
        }
    }

    pub fn draw<F: GpuFuture + 'static>(
        &mut self,
        future: F,
        final_image: Arc<dyn ImageViewAbstract + 'static>,
        delta_time: u128,
        input: &Input
    ) -> vulkano::command_buffer::CommandBufferExecFuture<
        vulkano::command_buffer::CommandBufferExecFuture<
            vulkano::command_buffer::CommandBufferExecFuture<F, PrimaryAutoCommandBuffer>,
            PrimaryAutoCommandBuffer,
        >,
        PrimaryAutoCommandBuffer,
    > {

        self.process_input(&input, delta_time);

        self.rotation_light += 0.005;
        {
            let scene_locked = self.scene.lock().unwrap();
            let light_rot = Matrix4::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                Rad(self.rotation_light as f32),
            ) * Point3::new(40.0, 50.0, -20.0).to_homogeneous();
            scene_locked.directional_lights[0].lock().unwrap().position =
                Point3::new(light_rot.x, light_rot.y, light_rot.z);
        }

        self.shadow_map_renderer.begin_render_pass();
        self.shadow_map_renderer.draw();
        let shadow_future = Some(self.shadow_map_renderer.end_render_pass(future));

        self.deferred_map_renderer.begin_render_pass();
        self.deferred_map_renderer.draw();
        let deferred_future = self
            .deferred_map_renderer
            .end_render_pass(shadow_future.unwrap());

        self.lighting_renderer
            .begin_render_pass(final_image.clone());
        self.lighting_renderer.draw(
            self.shadow_map_renderer.shadow_image.clone(),
            self.deferred_map_renderer.position_image.clone(),
            self.deferred_map_renderer.albedo_specular_image.clone(),
            self.deferred_map_renderer.normals_image.clone(),
        );
        let lighting_future = self.lighting_renderer.end_render_pass(deferred_future);
        lighting_future
    }

    fn process_input(&mut self, input_handler: &Input, delta_time: u128) {
        if input_handler.key_held(VirtualKeyCode::A) {
            self.process_keyboard(CameraMovement::Left, delta_time as f32);
        }
        if input_handler.key_held(VirtualKeyCode::W) {
            self.process_keyboard(CameraMovement::Forward, delta_time as f32);
        }
        if input_handler.key_held(VirtualKeyCode::D) {
            self.process_keyboard(CameraMovement::Right, delta_time as f32);
        }
        if input_handler.key_held(VirtualKeyCode::S) {
            self.process_keyboard(CameraMovement::Backward, delta_time as f32);
        }

        self.process_mouse_movement(input_handler.mouse_dx, input_handler.mouse_dy);
    }

    pub fn process_mouse_movement(&self, x_offset: f32, y_offset: f32) {
        self.scene
            .lock()
            .unwrap()
            .active_camera
            .process_mouse_movement(x_offset, y_offset);
    }

    pub fn process_keyboard(&mut self, direction: CameraMovement, delta_time: f32) {
        self.scene
            .lock()
            .unwrap()
            .active_camera
            .process_keyboard(direction, delta_time);
    }
}
