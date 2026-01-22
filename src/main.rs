use std::sync::Arc;

use anyhow::Context;
use wgpu::util::DeviceExt;
use winit::{application::ApplicationHandler, event::{ElementState, KeyEvent, WindowEvent}, event_loop::{self, ActiveEventLoop, EventLoop}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowAttributes, WindowId}};



#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    pos: [f32; 2],
}

impl Vertex {
    const ATTRS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![0 => Float32x2];
    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

struct State {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pipeline: wgpu::RenderPipeline,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl State {
    async fn new(window: Window) -> Self {
        let window = Arc::new(window);
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = unsafe {
            instance.create_surface(window.clone())
        }.expect("create_surface failed.");

        let adapter = instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            }
        )
        .await
        .expect("No suitable GPU adapters found on the system!");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::defaults(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    ..Default::default()
                }
            )
            .await
            .expect("Failed to create device and queue");

        let caps = surface.get_capabilities(&adapter);

        let format = caps.formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        
        let present_mode = if caps.present_modes.contains(
            &wgpu::PresentMode::Mailbox
        ) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let alpha_mode = caps.alpha_modes[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let base = [
            Vertex { pos: [-0.25, -0.25] },
            Vertex { pos: [ 0.25, -0.25] },
            Vertex { pos: [ 0.25,  0.25] },
            Vertex { pos: [-0.25,  0.25] },
        ];

        let base_idx: [u16; 6] = [0, 1, 2, 0, 2, 3];

        let rects: &[(f32, f32, f32, f32)] = &[
            (-0.6, -0.4, 0.18, 0.18),
            (-0.1, -0.2, 0.12, 0.22),
            ( 0.5, -0.3, 0.25, 0.10),
            (-0.2,  0.5, 0.15, 0.15),
            ( 0.6,  0.5, 0.10, 0.20),
        ];

        let mut vertices: Vec<Vertex> = Vec::with_capacity(rects.len() * 4);
        let mut indices: Vec<u16> = Vec::with_capacity(rects.len() * 6);

        for (i, &(cx, cy, hw, hh)) in rects.iter().enumerate() {
            for v in base {
                vertices.push(Vertex {
                    pos: [cx + v.pos[0] * hw, cy + v.pos[1] * hh],
                });
            }

            let off = (i * 4) as u16;
            indices.extend_from_slice(&[
                base_idx[0] + off, base_idx[1] + off, base_idx[2] + off,
                base_idx[3] + off, base_idx[4] + off, base_idx[5] + off
            ]);
        }

        let vertex_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("vertex_buf"),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }
        );

        let index_buf = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("index_buf"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::INDEX,
            }
        );

        let shader_src = r#"
            struct VSOut {
                @builtin(position) pos: vec4<f32>,
            };

            @vertex
            fn vs_main(@location(0) in_pos: vec2<f32>) -> VSOut {
                var out: VSOut;
                out.pos = vec4<f32>(in_pos, 0.0, 1.0);
                return out;
            }

            @fragment
            fn fs_main() -> @location(0) vec4<f32> {
                // 単色（緑）
                return vec4<f32>(0.1, 0.8, 0.2, 1.0);
            }
        "#;

        let shader = device.create_shader_module(
            wgpu::ShaderModuleDescriptor {
                label: Some("shader"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            }
        );

        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("pipeline_layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            }
        );

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
        }

    }

    fn resize(&mut self, w: u32, h: u32) {
        self.config.width = w.max(1);
        self.config.height = h.max(1);
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) {
        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Err(wgpu::SurfaceError::Outdated) => return,
            Err(e) => {
                eprintln!("Failed to acquire next swap chain texture: {:?}", e);
                return;
            }
        };

         let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });

        {
            let mut rpass = encoder.begin_render_pass(
                &wgpu::RenderPassDescriptor {
                    label: Some("render_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                wgpu::Color {
                                    r: 0.05,
                                    g: 0.05,
                                    b: 0.08,
                                    a: 1.0
                                }
                            ),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                }
            );

            rpass.set_pipeline(&self.pipeline);
            rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }

    
}

struct App {
    state: Option<State>,
    window_id: Option<WindowId>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            state: None,
            window_id: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(WindowAttributes::default().with_title("aaa"))
            .expect("create_window failed");

        self.window_id = Some(window.id());
        self.state = Some(
            pollster::block_on(State::new(window))
        );
    }

    fn window_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        id: WindowId,
        event: winit::event::WindowEvent,
    ) {
        if Some(id) != self.window_id {
            return;
        }
        let Some(state) = self.state.as_mut() else {
            return;
        };

        match event {
            WindowEvent::Resized(sz) => state.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => state.render(),
            WindowEvent::CloseRequested => std::process::exit(0),
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(KeyCode::Escape),
                    state: ElementState::Pressed,
                    ..
                },
                ..
            } => std::process::exit(0),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}


fn main() {
    let event_loop = EventLoop::new().expect("EventLoop::new failed");
    let mut app = App::default();
    event_loop.run_app(&mut app).expect("run_app failed");
}