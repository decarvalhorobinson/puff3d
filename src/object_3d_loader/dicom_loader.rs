use cgmath::Vector3;
use dicom::object::open_file;
use dicom::pixeldata::PixelDecoder;

use crate::scene_pkg::volume::Volume;


pub fn load_dicom_file_to_volume() -> Volume {

    let mut volume = Volume::new();
    for n in 1001..1246  {
        let  path = format!("./src/Head/vhm.{}.dcm", n);
        let obj = open_file(path).unwrap();
        let pixel_data = obj.decode_pixel_data().unwrap();
        let columns = pixel_data.columns();
        let rows: u32 = pixel_data.rows();

        let pixels: Vec<u16> = pixel_data.data_ow();
        volume.pixel_data.extend(pixels);
        volume.dimension[0] = rows;
        volume.dimension[1] = columns;
        volume.dimension[2] += 1;

        

    }
    for n in 0..267  {
    
        for n in 0..volume.dimension[0]*volume.dimension[1] {
            volume.pixel_data.push(0);
        }
        volume.dimension[2] += 1;
    }
    println!("resolution: {:?}", volume.dimension);
    //volume.transform.rotation = Vector3::new(90.0, 90.0, 00.0);
    volume.update_model_matrix();
    volume
}