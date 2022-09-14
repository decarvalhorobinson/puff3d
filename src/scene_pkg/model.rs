use super::mesh_model::Mesh;

pub struct Transform {
    pub position: [f32; 3],
    pub rotation: [f32; 3],
    pub scale: [f32; 3]
}

pub struct Material {
    pub diffuse: String,
    pub normal: String,
    pub specular: String,
    pub metalness: String,
    pub roughness: String
}

pub struct Object3D {
    pub transform: Transform,
    pub mesh: Option<Mesh>,
    pub material: Option<Material>,
    pub visibility: bool,
}


pub struct DirectionalLight {
    pub position: [f32; 3],
    pub direction: [f32; 3],
}

pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub right_direction: [f32; 3],
    pub world_up_direction: [f32; 3],
    pub pitch_yaw_roll: [f32; 3],
    pub last_position: [f32; 3],
    pub z_near: f32,
    pub z_far: f32,
    pub mouse_sensitivity: f32,
    pub movement_speed: f32,
}

pub struct Scene {
    pub cameras: Vec<Camera>,
    pub active_camera: Camera,
    pub objects: Vec<Object3D>,
    pub directional_lights: Vec<DirectionalLight>
}