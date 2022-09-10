use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use regex::Regex;
use std::io::prelude::*;

use crate::system::mesh::{Mesh, Vertex, Normal, Uv};


pub struct ObjFileToMeshConverter {
    path: String
}

impl ObjFileToMeshConverter {
    // static methods
    pub fn new(path: String) -> ObjFileToMeshConverter {
        ObjFileToMeshConverter {
            path
        }
    }
    //public methods
    pub fn create_mesh(&self) -> Mesh{
        let mesh: Mesh;

        let mut vertex_indices: Vec<u16> = vec![];
        let mut uv_indices: Vec<u16> = vec![];
        let mut normal_indices: Vec<u16> = vec![];

	    let mut tmp_vertices: Vec<Vertex> = vec![];
	    let mut tmp_normals: Vec<Normal> = vec![];
	    let mut tmp_uvs: Vec<Uv> = vec![];



        let data_iterator = read_lines(self.path.clone());
        if let Ok(lines) = data_iterator{
            for line_result in lines {
                if let Ok(line) = line_result {
                    let mut line_parts = line.split(" ");
                    let line_id = line_parts.next().unwrap();
                    match line_id {
                        "v" => {
                            let x = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let y = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let z = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let vertex = Vertex { position: [x, y, z] };
                            tmp_vertices.push(vertex);
                         
                        }
                        "vt" => {
                            let u = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let v = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let uv = Uv { uv: [u, v] };
                            tmp_uvs.push(uv);
                         
                        }
                        "vn" => {
                            let x = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let y = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let z = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let normal = Normal { normal: [x, y, z] };
                            tmp_normals.push(normal);
                        }
                        "f" => {
                            let mut vertex_index: [u16; 3] = [0, 0, 0];
                            let mut uv_index: [u16; 3] = [0, 0, 0];
                            let mut normal_index: [u16; 3] = [0, 0, 0];

                            let mut indexString = line_parts.next().unwrap().split("/");
                            vertex_index[0] = indexString.next().unwrap().parse::<u16>().unwrap();
                            uv_index[0] = indexString.next().unwrap().parse::<u16>().unwrap();
                            normal_index[0] = indexString.next().unwrap().parse::<u16>().unwrap();

                            indexString = line_parts.next().unwrap().split("/");
                            vertex_index[1] = indexString.next().unwrap().parse::<u16>().unwrap();
                            uv_index[1] = indexString.next().unwrap().parse::<u16>().unwrap();;
                            normal_index[1] = indexString.next().unwrap().parse::<u16>().unwrap();

                            indexString = line_parts.next().unwrap().split("/");
                            vertex_index[2] = indexString.next().unwrap().parse::<u16>().unwrap();
                            uv_index[2] = indexString.next().unwrap().parse::<u16>().unwrap();
                            normal_index[2] = indexString.next().unwrap().parse::<u16>().unwrap();

                            vertex_indices.push(vertex_index[0]);
                            vertex_indices.push(vertex_index[1]);
                            vertex_indices.push(vertex_index[2]);

                            uv_indices.push(uv_index[0]);
                            uv_indices.push(uv_index[1]);
                            uv_indices.push(uv_index[2]);

                            normal_indices.push(normal_index[0]);
                            normal_indices.push(normal_index[1]);
                            normal_indices.push(normal_index[2]);
                            
                        }
                        _ => {}
                    }
                }
            }
        }

        let mut mesh: Mesh = Mesh { vertices: vec![], normals: vec![], uvs: vec![], indices: vec![] };
        for i in 0..vertex_indices.len() {

           let vertex_index: usize = vertex_indices[i] as usize;
           mesh.vertices.push(tmp_vertices[vertex_index - 1]); 

           let normal_index: usize = normal_indices[i] as usize;
           mesh.normals.push(tmp_normals[normal_index - 1]);

           let uv_index: usize = uv_indices[i] as usize;
           mesh.uvs.push(tmp_uvs[uv_index - 1]);

           mesh.indices.push(i as u16);

        }

        return mesh;



    }

}


fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}