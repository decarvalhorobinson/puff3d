use cgmath::Vector3;

#[repr(C)]
#[derive(Clone, Debug, Copy)]
pub struct Transform {
    pub position: Vector3<f32>,
    pub rotation: Vector3<f32>,
    pub scale: Vector3<f32>,
}
