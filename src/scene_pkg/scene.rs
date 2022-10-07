use std::sync::{Arc, Mutex};

use cgmath::{Matrix4, Rad, Vector3};

use super::{camera::Camera, directional_light::DirectionalLight, object3d::Object3D, volume::Volume};

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Scene {
    pub cameras: Vec<Camera>,
    pub active_camera: Camera,
    pub objects: Vec<Object3D>,
    pub volumes: Vec<Volume>,
    pub world_model: Matrix4<f32>,
    pub directional_lights: Vec<Arc<Mutex<DirectionalLight>>>,
}

impl Scene {
    pub fn projection(viewport_dimensions: [u32; 2]) -> Matrix4<f32> {
        let aspect_ratio = viewport_dimensions[0] as f32 / viewport_dimensions[1] as f32;
        cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.1, 70.0)
    }

    pub fn rotate(&mut self, rotation: f32) {
        self.world_model =
            Matrix4::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), Rad(rotation as f32));
    }
}
