use super::{camera::Camera, object3d::Object3D, directional_ligh::DirectionalLight};

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Scene {
    pub cameras: Vec<Camera>,
    pub active_camera: Camera,
    pub objects: Vec<Object3D>,
    pub directional_lights: Vec<DirectionalLight>
}