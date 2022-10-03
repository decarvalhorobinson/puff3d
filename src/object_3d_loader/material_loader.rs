use image::io::Reader as ImageReader;

pub fn load_material(image_path: String) {
    let img = ImageReader::open(image_path).unwrap().decode().unwrap();
    let a = img.into_bytes();
}