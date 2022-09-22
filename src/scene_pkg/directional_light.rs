use cgmath::{EuclideanSpace, Matrix4, Point3, Vector3};

#[repr(C)]
#[derive(Clone, Debug)]
pub struct DirectionalLight {
    pub position: Point3<f32>,
    pub center: Point3<f32>,
    pub color: [f32; 4],
}

impl DirectionalLight {
    pub fn view_projection(self) -> (Matrix4<f32>, Matrix4<f32>) {
        let near_plane: f32 = 0.0f32;
        let far_plane: f32 = 90.5f32;
        let projection = cgmath::ortho(-40.0f32, 40.0f32, -40.0f32, 40.0f32, near_plane, far_plane);
        let view: Matrix4<f32> =
            Matrix4::look_at_rh(self.position, self.center, Vector3::new(0.0, -1.0, 0.0));
        (view, projection)
    }

    pub fn direction(self) -> Vector3<f32> {
        self.center.to_vec() - self.position.to_vec()
    }
}
