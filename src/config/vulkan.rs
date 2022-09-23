use std::sync::Arc;
use std::time::Instant;

use vulkano::VulkanLibrary;
use vulkano::device::physical::{PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::ImageUsage;
use vulkano::instance::{Instance, InstanceCreateInfo};

use vulkano::swapchain::{
    acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo,
    SwapchainCreationError, PresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano_win::VkSurfaceBuild;

use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use super::mesh_example;
use crate::frame::scene_renderer::SceneRenderer;

fn create_instance() -> Arc<Instance> {
    let library = VulkanLibrary::new().unwrap();
    let required_extensions = vulkano_win::required_extensions(&library);
    Instance::new(
        library,
        InstanceCreateInfo {
        enabled_extensions: required_extensions,
        // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
        enumerate_portability: true,
        ..Default::default()
    })
    .unwrap()
}

fn create_surface(instance: Arc<Instance>) -> (Arc<Surface<winit::window::Window>>, EventLoop<()>) {
    let event_loop = EventLoop::new();
    (
        WindowBuilder::new()
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap(),
        event_loop,
    )
}

fn create_device<'a>(
    surface: Arc<vulkano::swapchain::Surface<winit::window::Window>>,
    instance: Arc<Instance>,
) -> (Arc<Device>, Arc<vulkano::device::Queue>) {
    // select best physical device
    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };
    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.graphics && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
            _ => 5,
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
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();
    (device, queues.next().unwrap())
}

fn create_swapchain(
    device: Arc<Device>,
    surface: Arc<vulkano::swapchain::Surface<winit::window::Window>>,
) -> (
    Arc<Swapchain<winit::window::Window>>,
    Vec<Arc<ImageView<vulkano::image::SwapchainImage<winit::window::Window>>>>,
) {
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(&surface, Default::default())
        .unwrap();
    let image_format = Some(
        device
            .physical_device()
            .surface_formats(&surface, Default::default())
            .unwrap()[0]
            .0,
    );

    let (swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count + 1,
            image_format,
            image_extent: surface.window().inner_size().into(),
            image_usage: ImageUsage {
                color_attachment: true,
                ..ImageUsage::empty()
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

    //create scene, should be unique for everything
    let scene = mesh_example::get_example_scene_cottage_house();

    // Renderer for the scene
    let mut scene_renderer = SceneRenderer::new(queue.clone(), scene.clone(), swapchain.image_format());

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());


    let mut last_frame = Instant::now();

    let mut fps = 0.0;
    let mut counter = 0;
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
            // calculate delta time
            let current_frame = Instant::now();
            let delta_time = current_frame.checked_duration_since(last_frame).unwrap().as_millis();
            last_frame = current_frame;


            if counter > 100 {
                let fps_average = fps/counter as f32;
                println!("{:?}", 1000.0/fps_average as f32);
                counter = 0;
                fps = 0.0;
                
                
            }else{
                counter += 1;
                fps += delta_time as f32;
            }
            


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

            let future = scene_renderer.draw(future, images[image_num].clone(), delta_time)
                .then_swapchain_present(
                    queue.clone(),
                    PresentInfo {
                        index: image_num,
                        ..PresentInfo::swapchain(swapchain.clone())
                    },
                )
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
