use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

use crate::scene_pkg::mesh::{Mesh, Normal, Tangent, Uv, Vertex};

pub struct ObjFileToMeshConverter {
    path: String,
}

impl ObjFileToMeshConverter {
    // static methods
    pub fn new(path: String) -> ObjFileToMeshConverter {
        ObjFileToMeshConverter { path }
    }
    //public methods
    pub fn create_mesh(&self) -> Mesh {
        let mut vertex_indices: Vec<u16> = vec![];
        let mut uv_indices: Vec<u16> = vec![];
        let mut normal_indices: Vec<u16> = vec![];

        let mut tmp_vertices: Vec<Vertex> = vec![];
        let mut tmp_normals: Vec<Normal> = vec![];
        let mut tmp_uvs: Vec<Uv> = vec![];

        let data_iterator = read_lines(self.path.clone());
        if let Ok(lines) = data_iterator {
            for line_result in lines {
                if let Ok(line) = line_result {
                    let mut line_parts = line.split(" ");
                    let line_id = line_parts.next().unwrap();
                    match line_id {
                        "v" => {
                            let x = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let y = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let z = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let vertex = Vertex {
                                position: [x, y, z],
                            };
                            tmp_vertices.push(vertex);
                        }
                        "vt" => {
                            let u = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let v = line_parts.next().unwrap().parse::<f32>().unwrap();
                            let uv = Uv { uv: [u, -v] };
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

                            let mut index_string = line_parts.next().unwrap().split("/");
                            vertex_index[0] = index_string.next().unwrap().parse::<u16>().unwrap();
                            uv_index[0] = index_string.next().unwrap().parse::<u16>().unwrap();
                            normal_index[0] = index_string.next().unwrap().parse::<u16>().unwrap();

                            index_string = line_parts.next().unwrap().split("/");
                            vertex_index[1] = index_string.next().unwrap().parse::<u16>().unwrap();
                            uv_index[1] = index_string.next().unwrap().parse::<u16>().unwrap();
                            normal_index[1] = index_string.next().unwrap().parse::<u16>().unwrap();

                            index_string = line_parts.next().unwrap().split("/");
                            vertex_index[2] = index_string.next().unwrap().parse::<u16>().unwrap();
                            uv_index[2] = index_string.next().unwrap().parse::<u16>().unwrap();
                            normal_index[2] = index_string.next().unwrap().parse::<u16>().unwrap();

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

        let mut mesh: Mesh = Mesh {
            vertices: vec![],
            normals: vec![],
            uvs: vec![],
            tangent: vec![],
            indices: vec![],
        };
        for i in 0..vertex_indices.len() {
            let vertex_index: usize = vertex_indices[i] as usize;
            mesh.vertices.push(tmp_vertices[vertex_index - 1]);

            let normal_index: usize = normal_indices[i] as usize;
            mesh.normals.push(tmp_normals[normal_index - 1]);

            let uv_index: usize = uv_indices[i] as usize;
            mesh.uvs.push(tmp_uvs[uv_index - 1]);

            mesh.indices.push(i as u16);
        }

        // calculate tangent and bitangent after indices have beeen reorganized
        for i in (0..mesh.vertices.len()).step_by(3) {
            // Shortcuts for vertices
            let v0 = mesh.vertices[i + 0];
            let v1 = mesh.vertices[i + 1];
            let v2 = mesh.vertices[i + 2];

            // Shortcuts for UVs
            let uv0 = mesh.uvs[i + 0];
            let uv1 = mesh.uvs[i + 1];
            let uv2 = mesh.uvs[i + 2];

            // Edges of the triangle : position delta
            let delta_pos1 = v1 - v0;
            let delta_pos2 = v2 - v0;

            let delta_uv1 = uv1 - uv0;
            let delta_uv2 = uv2 - uv0;

            let r: f32 =
                1.0f32 / (delta_uv1.uv[0] * delta_uv2.uv[1] - delta_uv1.uv[1] * delta_uv2.uv[0]);
            let tangent = (delta_pos1 * delta_uv2.uv[1] - delta_pos2 * delta_uv1.uv[1]) * r;

            mesh.tangent.push(Tangent {
                tangent: tangent.position,
            });
            mesh.tangent.push(Tangent {
                tangent: tangent.position,
            });
            mesh.tangent.push(Tangent {
                tangent: tangent.position,
            });
        }

        return mesh;
    }
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}
