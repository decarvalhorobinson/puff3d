#[repr(C)]
#[derive(Clone, Debug)]
pub struct DirectionalLight {
    pub direction: [f32; 3],
    pub color: [f32; 3],
}

