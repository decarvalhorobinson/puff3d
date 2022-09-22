use cgmath::{Matrix4, Point3, Vector3};

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up_direction: Vector3<f32>,
}

impl Camera {
    pub fn new() -> Camera {
        Camera {
            position: Point3::new(-40.0, 20.0, 20.0),
            target: Point3::new(0.0f32, 0.0f32, 0.0f32),
            up_direction: Vector3::new(0.0f32, -1.0f32, 0.0f32),
        }
    }

    pub fn get_view_matrix(&self) -> Matrix4<f32> {
        let view = Matrix4::look_at_rh(self.position, self.target, self.up_direction);
        let scale = Matrix4::from_scale(1.0);
        view * scale
    }
}
