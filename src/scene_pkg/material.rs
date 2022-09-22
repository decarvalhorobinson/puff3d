#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct Material {
    pub diffuse_file_path: String,
    pub normal_file_path: String,
    pub specular_file_path: String,
    pub metalness_file_path: String,
    pub roughness_file_path: String,
}
