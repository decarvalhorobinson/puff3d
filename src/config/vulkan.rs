use std::sync::Arc;

use cgmath::{Matrix4, SquareMatrix, Vector3};
use vulkano::device::{Device, DeviceExtensions, DeviceCreateInfo, QueueCreateInfo};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::image::ImageUsage;
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateInfo};

use vulkano::swapchain::{Swapchain, SwapchainCreateInfo, Surface, SwapchainCreationError, acquire_next_image, AcquireError};
use vulkano::sync::{self, GpuFuture, FlushError};
use vulkano_win::VkSurfaceBuild;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use crate::frame::Pass;
use crate::frame::FrameSystem;
use crate::system::{mesh_draw_system::MeshDrawSystem};

use super::mesh_example;


fn create_instance() -> Arc<Instance> {
     
     let required_extensions = vulkano_win::required_extensions();
     Instance::new(
         InstanceCreateInfo {
             enabled_extensions: required_extensions,
             // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
             enumerate_portability: true,
             ..Default::default()
         },
     )
     .unwrap()
}

fn create_surface(instance: Arc<Instance>) -> (Arc<Surface<winit::window::Window>>, EventLoop<()>) {
    let event_loop = EventLoop::new();
    (WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap(), event_loop)
}

fn create_device<'a>(
    surface: Arc<vulkano::swapchain::Surface<winit::window::Window>>, 
    instance: Arc<Instance>
) -> (Arc<Device>, Arc<vulkano::device::Queue>)  {

    // select best physical device
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();
    

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type
    );

    // create the device and a queue
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();
    (device, queues.next().unwrap())

}

fn create_swapchain(
    device: Arc<Device>, 
    surface: Arc<vulkano::swapchain::Surface<winit::window::Window>>
) -> 
(
    Arc<Swapchain<winit::window::Window>>, 
    Vec<Arc<ImageView<vulkano::image::SwapchainImage<winit::window::Window>>>>
) {
    let surface_capabilities = device.physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();
    let image_format = Some(
        device.physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count,
            image_format,
            image_extent: surface.window().inner_size().into(),
            image_usage: ImageUsage {
                color_attachment: true,
                ..ImageUsage::none()
            },
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .iter()
                .next()
                .unwrap(),
            ..Default::default()
        },
    )
    .unwrap();
    let images = images
        .into_iter()
        .map(|image| ImageView::new_default(image).unwrap())
        .collect::<Vec<_>>();
    (swapchain, images)
}

pub fn vulkan_init() {
   
    // create instance
    let instance = create_instance();

    // create windows and surface to draw on
    
    let (surface, event_loop) = create_surface(instance.clone());

    // create device and queue
    let (device, queue) = create_device(surface.clone(), instance.clone());
    
    // create swapchain and the images
    let (mut swapchain, mut images) = create_swapchain(device.clone(), surface.clone());



    // Here is the basic initialization for the deferred system.
    let mut frame_system = FrameSystem::new(queue.clone(), swapchain.image_format());
    let mesh_draw_system = MeshDrawSystem::new(queue.clone(), frame_system.deferred_subpass(), mesh_example::get_example_mesh2());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            let dimensions = surface.window().inner_size();
            if dimensions.width == 0 || dimensions.height == 0 {
                return;
            }

            previous_frame_end.as_mut().unwrap().cleanup_finished();

            if recreate_swapchain {
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: dimensions.into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };
                let new_images = new_images
                    .into_iter()
                    .map(|image| ImageView::new_default(image).unwrap())
                    .collect::<Vec<_>>();

                swapchain = new_swapchain;
                images = new_images;
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let future = previous_frame_end.take().unwrap().join(acquire_future);

            let mut frame = frame_system.frame(future, images[image_num].clone(), Matrix4::identity());
            let mut after_future = None;
            while let Some(pass) = frame.next_pass() {
                match pass {
                    Pass::Deferred(mut draw_pass) => {
                        let cb = mesh_draw_system.draw(draw_pass.viewport_dimensions(), draw_pass.world_to_framebuffer_matrix());
                        draw_pass.execute(cb);
                    }
                    Pass::Lighting(mut lighting) => {
                        lighting.ambient_light([0.2, 0.2, 0.2]);
                        lighting.directional_light(Vector3::new(0.2, -0.1, -0.7), [0.6, 0.6, 0.6]);
                        lighting.point_light(Vector3::new(0.5, -0.5, -0.1), [1.0, 0.0, 0.0]);
                        lighting.point_light(Vector3::new(-0.9, 0.2, -0.15), [0.0, 1.0, 0.0]);
                        lighting.point_light(Vector3::new(0.0, 0.5, -0.05), [0.0, 0.0, 1.0]);
                    }
                    Pass::Finished(af) => {
                        after_future = Some(af);
                    }
                }
            }

            let future = after_future
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(sync::now(device.clone()).boxed());
                }
            }
        }
        _ => (),
    });

}