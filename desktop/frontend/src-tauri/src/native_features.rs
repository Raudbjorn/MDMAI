use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use tauri::{
    AppHandle, Manager, Runtime,
    tray::{TrayIcon, TrayIconBuilder},
    menu::{Menu, MenuItem, Submenu},
};
use log::{info, error, warn, debug};

use crate::process_manager::{ProcessManagerState, ProcessState, HealthStatus};

/// System tray manager for TTRPG Assistant
pub struct SystemTrayManager {
    app_handle: Arc<Mutex<Option<AppHandle>>>,
    tray_icon: Arc<Mutex<Option<TrayIcon>>>,
    menu_items: Arc<RwLock<TrayMenuItems>>,
}

/// Tray menu item identifiers
#[derive(Debug, Clone)]
struct TrayMenuItems {
    show_hide: String,
    mcp_server_status: String,
    start_server: String,
    stop_server: String,
    restart_server: String,
    quick_campaign: String,
    quick_rulebook: String,
    settings: String,
    about: String,
    quit: String,
}

impl Default for TrayMenuItems {
    fn default() -> Self {
        TrayMenuItems {
            show_hide: "show_hide".to_string(),
            mcp_server_status: "mcp_status".to_string(),
            start_server: "start_server".to_string(),
            stop_server: "stop_server".to_string(),
            restart_server: "restart_server".to_string(),
            quick_campaign: "quick_campaign".to_string(),
            quick_rulebook: "quick_rulebook".to_string(),
            settings: "settings".to_string(),
            about: "about".to_string(),
            quit: "quit".to_string(),
        }
    }
}

/// File association configuration for TTRPG file types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileAssociation {
    pub extension: String,
    pub mime_type: String,
    pub description: String,
    pub icon_path: Option<String>,
    pub is_default: bool,
}

/// Drag and drop event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DragDropEvent {
    pub event_type: String,
    pub files: Vec<String>,
    pub position: Option<(f64, f64)>,
    pub timestamp: u64,
}

/// Native file dialog options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDialogOptions {
    pub title: String,
    pub default_path: Option<String>,
    pub filters: Vec<FileDialogFilter>,
    pub multiple: bool,
    pub directory: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileDialogFilter {
    pub name: String,
    pub extensions: Vec<String>,
}

/// OS notification data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationData {
    pub title: String,
    pub body: String,
    pub icon: Option<String>,
    pub sound: bool,
    pub urgency: NotificationUrgency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationUrgency {
    Low,
    Normal,
    Critical,
}

impl SystemTrayManager {
    /// Create a new system tray manager
    pub fn new() -> Self {
        SystemTrayManager {
            app_handle: Arc::new(Mutex::new(None)),
            tray_icon: Arc::new(Mutex::new(None)),
            menu_items: Arc::new(RwLock::new(TrayMenuItems::default())),
        }
    }

    /// Initialize the system tray with TTRPG-specific menu
    pub async fn initialize<R: Runtime>(&self, app: &AppHandle<R>) -> Result<(), Box<dyn std::error::Error>> {
        *self.app_handle.lock().await = Some(app.clone());
        
        // Build the tray menu
        let menu = self.build_tray_menu().await?;
        
        // Create tray icon
        let tray = TrayIconBuilder::with_id("main")
            .menu(&menu)
            .tooltip("TTRPG Assistant")
            .on_menu_event({
                let app_clone = app.clone();
                let menu_items = self.menu_items.clone();
                move |_tray, event| {
                    let app_clone = app_clone.clone();
                    let menu_items = menu_items.clone();
                    tauri::async_runtime::spawn(async move {
                        SystemTrayManager::handle_tray_menu_event_static(app_clone, event, menu_items).await;
                    });
                }
            })
            .build(app)?;
        
        *self.tray_icon.lock().await = Some(tray);
        
        info!("System tray initialized successfully");
        Ok(())
    }

    /// Build the system tray menu with TTRPG-specific items
    async fn build_tray_menu(&self) -> Result<Menu<tauri::Wry>, Box<dyn std::error::Error>> {
        let menu_items = self.menu_items.read().await;
        
        let menu = Menu::with_items(&[
            &MenuItem::with_id(&menu_items.show_hide, "Show/Hide Window", true, None::<&str>)?,
            &MenuItem::separator()?,
            
            // MCP Server Management Section
            &MenuItem::with_id(&menu_items.mcp_server_status, "MCP Server: Stopped", false, None::<&str>)?,
            &MenuItem::with_id(&menu_items.start_server, "Start Server", true, None::<&str>)?,
            &MenuItem::with_id(&menu_items.stop_server, "Stop Server", false, None::<&str>)?,
            &MenuItem::with_id(&menu_items.restart_server, "Restart Server", false, None::<&str>)?,
            &MenuItem::separator()?,
            
            // Quick Actions Section
            &Submenu::with_id_and_items(
                "quick_actions",
                "Quick Actions",
                true,
                &[
                    &MenuItem::with_id(&menu_items.quick_campaign, "New Campaign", true, None::<&str>)?,
                    &MenuItem::with_id(&menu_items.quick_rulebook, "Import Rulebook", true, None::<&str>)?,
                ]
            )?,
            &MenuItem::separator()?,
            
            // Settings and About
            &MenuItem::with_id(&menu_items.settings, "Settings", true, None::<&str>)?,
            &MenuItem::with_id(&menu_items.about, "About", true, None::<&str>)?,
            &MenuItem::separator()?,
            &MenuItem::with_id(&menu_items.quit, "Quit", true, None::<&str>)?,
        ])?;
        
        Ok(menu)
    }

    /// Handle tray menu events (static version for async callback)
    async fn handle_tray_menu_event_static<R: Runtime>(
        app: AppHandle<R>, 
        event: tauri::menu::MenuEvent,
        menu_items: Arc<RwLock<TrayMenuItems>>
    ) {
        let menu_items = menu_items.read().await;
        
        match event.id().as_ref() {
            id if id == &menu_items.show_hide => {
                Self::toggle_window_visibility(&app).await;
            },
            id if id == &menu_items.start_server => {
                Self::handle_start_server(&app).await;
            },
            id if id == &menu_items.stop_server => {
                Self::handle_stop_server(&app).await;
            },
            id if id == &menu_items.restart_server => {
                Self::handle_restart_server(&app).await;
            },
            id if id == &menu_items.quick_campaign => {
                Self::handle_quick_campaign(&app).await;
            },
            id if id == &menu_items.quick_rulebook => {
                Self::handle_quick_rulebook(&app).await;
            },
            id if id == &menu_items.settings => {
                Self::handle_settings(&app).await;
            },
            id if id == &menu_items.about => {
                Self::handle_about(&app).await;
            },
            id if id == &menu_items.quit => {
                Self::handle_quit(&app).await;
            },
            _ => {
                warn!("Unknown tray menu item clicked: {}", event.id());
            }
        }
    }

    /// Toggle main window visibility
    async fn toggle_window_visibility<R: Runtime>(app: &AppHandle<R>) {
        if let Some(window) = app.get_webview_window("main") {
            match window.is_visible() {
                Ok(true) => {
                    if let Err(e) = window.hide() {
                        error!("Failed to hide window: {}", e);
                    }
                },
                Ok(false) => {
                    if let Err(e) = window.show() {
                        error!("Failed to show window: {}", e);
                    }
                    if let Err(e) = window.set_focus() {
                        error!("Failed to focus window: {}", e);
                    }
                },
                Err(e) => {
                    error!("Failed to check window visibility: {}", e);
                }
            }
        }
    }

    /// Handle start server action from tray
    async fn handle_start_server<R: Runtime>(app: &AppHandle<R>) {
        if let Err(e) = app.emit("tray-action", "start_server") {
            error!("Failed to emit start server event: {}", e);
        }
    }

    /// Handle stop server action from tray
    async fn handle_stop_server<R: Runtime>(app: &AppHandle<R>) {
        if let Err(e) = app.emit("tray-action", "stop_server") {
            error!("Failed to emit stop server event: {}", e);
        }
    }

    /// Handle restart server action from tray
    async fn handle_restart_server<R: Runtime>(app: &AppHandle<R>) {
        if let Err(e) = app.emit("tray-action", "restart_server") {
            error!("Failed to emit restart server event: {}", e);
        }
    }

    /// Handle quick campaign creation
    async fn handle_quick_campaign<R: Runtime>(app: &AppHandle<R>) {
        if let Err(e) = app.emit("tray-action", "quick_campaign") {
            error!("Failed to emit quick campaign event: {}", e);
        }
    }

    /// Handle quick rulebook import
    async fn handle_quick_rulebook<R: Runtime>(app: &AppHandle<R>) {
        if let Err(e) = app.emit("tray-action", "quick_rulebook") {
            error!("Failed to emit quick rulebook event: {}", e);
        }
    }

    /// Handle settings action
    async fn handle_settings<R: Runtime>(app: &AppHandle<R>) {
        if let Err(e) = app.emit("tray-action", "settings") {
            error!("Failed to emit settings event: {}", e);
        }
    }

    /// Handle about action
    async fn handle_about<R: Runtime>(app: &AppHandle<R>) {
        if let Err(e) = app.emit("tray-action", "about") {
            error!("Failed to emit about event: {}", e);
        }
    }

    /// Handle quit action
    async fn handle_quit<R: Runtime>(app: &AppHandle<R>) {
        info!("Quit requested from system tray");
        app.exit(0);
    }

    /// Update tray menu based on MCP server status
    pub async fn update_server_status(&self, state: ProcessState, health: HealthStatus) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(app) = self.app_handle.lock().await.as_ref() {
            if let Some(tray) = self.tray_icon.lock().await.as_ref() {
                let menu_items = self.menu_items.read().await;
                
                // Update status text and menu item availability
                let (status_text, start_enabled, stop_enabled, restart_enabled) = match state {
                    ProcessState::Stopped => ("MCP Server: Stopped", true, false, false),
                    ProcessState::Starting => ("MCP Server: Starting...", false, false, false),
                    ProcessState::Running => {
                        let health_text = match health {
                            HealthStatus::Healthy => "Running (Healthy)",
                            HealthStatus::Degraded => "Running (Degraded)",
                            HealthStatus::Unhealthy => "Running (Unhealthy)",
                            HealthStatus::Unknown => "Running",
                        };
                        (format!("MCP Server: {}", health_text), false, true, true)
                    },
                    ProcessState::Stopping => ("MCP Server: Stopping...", false, false, false),
                    ProcessState::Crashed => ("MCP Server: Crashed", true, false, false),
                    ProcessState::Restarting => ("MCP Server: Restarting...", false, false, false),
                };

                // Rebuild menu with updated status
                let menu = Menu::with_items(&[
                    &MenuItem::with_id(&menu_items.show_hide, "Show/Hide Window", true, None::<&str>)?,
                    &MenuItem::separator()?,
                    &MenuItem::with_id(&menu_items.mcp_server_status, status_text, false, None::<&str>)?,
                    &MenuItem::with_id(&menu_items.start_server, "Start Server", start_enabled, None::<&str>)?,
                    &MenuItem::with_id(&menu_items.stop_server, "Stop Server", stop_enabled, None::<&str>)?,
                    &MenuItem::with_id(&menu_items.restart_server, "Restart Server", restart_enabled, None::<&str>)?,
                    &MenuItem::separator()?,
                    &Submenu::with_id_and_items(
                        "quick_actions",
                        "Quick Actions",
                        true,
                        &[
                            &MenuItem::with_id(&menu_items.quick_campaign, "New Campaign", true, None::<&str>)?,
                            &MenuItem::with_id(&menu_items.quick_rulebook, "Import Rulebook", true, None::<&str>)?,
                        ]
                    )?,
                    &MenuItem::separator()?,
                    &MenuItem::with_id(&menu_items.settings, "Settings", true, None::<&str>)?,
                    &MenuItem::with_id(&menu_items.about, "About", true, None::<&str>)?,
                    &MenuItem::separator()?,
                    &MenuItem::with_id(&menu_items.quit, "Quit", true, None::<&str>)?,
                ])?;

                tray.set_menu(Some(menu))?;
                
                // Update tray icon based on status
                let icon_name = match (state, health) {
                    (ProcessState::Running, HealthStatus::Healthy) => "icon-green",
                    (ProcessState::Running, HealthStatus::Degraded) => "icon-yellow",
                    (ProcessState::Running, HealthStatus::Unhealthy) => "icon-red",
                    (ProcessState::Crashed, _) => "icon-red",
                    _ => "icon-gray",
                };
                
                // Update tooltip
                let tooltip = format!("TTRPG Assistant - {}", status_text);
                tray.set_tooltip(Some(&tooltip))?;
            }
        }
        
        Ok(())
    }
}

/// File association manager for TTRPG file types
pub struct FileAssociationManager {
    associations: Arc<RwLock<Vec<FileAssociation>>>,
}

impl FileAssociationManager {
    pub fn new() -> Self {
        FileAssociationManager {
            associations: Arc::new(RwLock::new(Self::default_associations())),
        }
    }

    /// Get default TTRPG file associations
    fn default_associations() -> Vec<FileAssociation> {
        vec![
            FileAssociation {
                extension: "pdf".to_string(),
                mime_type: "application/pdf".to_string(),
                description: "TTRPG Rulebook (PDF)".to_string(),
                icon_path: Some("icons/pdf-icon.png".to_string()),
                is_default: false,
            },
            FileAssociation {
                extension: "json".to_string(),
                mime_type: "application/json".to_string(),
                description: "TTRPG Campaign Data".to_string(),
                icon_path: Some("icons/campaign-icon.png".to_string()),
                is_default: true,
            },
            FileAssociation {
                extension: "yaml".to_string(),
                mime_type: "application/x-yaml".to_string(),
                description: "TTRPG Configuration".to_string(),
                icon_path: Some("icons/config-icon.png".to_string()),
                is_default: true,
            },
            FileAssociation {
                extension: "yml".to_string(),
                mime_type: "application/x-yaml".to_string(),
                description: "TTRPG Configuration".to_string(),
                icon_path: Some("icons/config-icon.png".to_string()),
                is_default: true,
            },
            FileAssociation {
                extension: "toml".to_string(),
                mime_type: "application/toml".to_string(),
                description: "TTRPG Settings".to_string(),
                icon_path: Some("icons/settings-icon.png".to_string()),
                is_default: true,
            },
            FileAssociation {
                extension: "character".to_string(),
                mime_type: "application/x-ttrpg-character".to_string(),
                description: "TTRPG Character Sheet".to_string(),
                icon_path: Some("icons/character-icon.png".to_string()),
                is_default: true,
            },
        ]
    }

    /// Register file associations with the operating system
    pub async fn register_associations(&self) -> Result<(), Box<dyn std::error::Error>> {
        let associations = self.associations.read().await;
        
        for association in associations.iter() {
            self.register_single_association(association).await?;
        }
        
        info!("File associations registered successfully");
        Ok(())
    }

    /// Register a single file association
    async fn register_single_association(&self, association: &FileAssociation) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(target_os = "windows")]
        {
            self.register_windows_association(association).await
        }
        
        #[cfg(target_os = "macos")]
        {
            self.register_macos_association(association).await
        }
        
        #[cfg(target_os = "linux")]
        {
            self.register_linux_association(association).await
        }
    }

    #[cfg(target_os = "windows")]
    async fn register_windows_association(&self, association: &FileAssociation) -> Result<(), Box<dyn std::error::Error>> {
        use std::process::Command;
        
        let exe_path = std::env::current_exe()?;
        let exe_str = exe_path.to_string_lossy();
        
        // Register file extension
        let reg_cmd = format!(
            r#"reg add "HKCU\Software\Classes\.{}" /ve /d "TTRPGAssistant.{}" /f"#,
            association.extension, association.extension
        );
        
        Command::new("cmd")
            .args(&["/C", &reg_cmd])
            .output()?;
        
        // Register application
        let app_cmd = format!(
            r#"reg add "HKCU\Software\Classes\TTRPGAssistant.{}" /ve /d "{}" /f"#,
            association.extension, association.description
        );
        
        Command::new("cmd")
            .args(&["/C", &app_cmd])
            .output()?;
        
        // Register command
        let cmd_reg = format!(
            r#"reg add "HKCU\Software\Classes\TTRPGAssistant.{}\shell\open\command" /ve /d "\"{}\" \"%1\"" /f"#,
            association.extension, exe_str
        );
        
        Command::new("cmd")
            .args(&["/C", &cmd_reg])
            .output()?;
        
        Ok(())
    }

    #[cfg(target_os = "macos")]
    async fn register_macos_association(&self, association: &FileAssociation) -> Result<(), Box<dyn std::error::Error>> {
        // macOS file associations are typically handled through Info.plist
        // This would require updating the bundle configuration
        warn!("macOS file association registration requires bundle configuration");
        Ok(())
    }

    #[cfg(target_os = "linux")]
    async fn register_linux_association(&self, association: &FileAssociation) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs;
        use std::path::PathBuf;
        
        let home = std::env::var("HOME")?;
        let desktop_file_path = PathBuf::from(&home)
            .join(".local/share/applications/ttrpg-assistant.desktop");
        
        let desktop_content = format!(
            r#"[Desktop Entry]
Name=TTRPG Assistant
Comment=Tabletop RPG Assistant
Exec=ttrpg-assistant %f
Icon=ttrpg-assistant
Terminal=false
Type=Application
Categories=Game;
MimeType={};
"#,
            association.mime_type
        );
        
        if let Some(parent) = desktop_file_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        fs::write(desktop_file_path, desktop_content)?;
        
        // Update MIME database
        std::process::Command::new("update-desktop-database")
            .arg(format!("{}/.local/share/applications", home))
            .output()
            .ok(); // Ignore errors, might not be available
        
        Ok(())
    }
}

/// Native features state manager
pub struct NativeFeaturesState {
    pub system_tray: Arc<SystemTrayManager>,
    pub file_associations: Arc<FileAssociationManager>,
}

impl NativeFeaturesState {
    pub fn new() -> Self {
        NativeFeaturesState {
            system_tray: Arc::new(SystemTrayManager::new()),
            file_associations: Arc::new(FileAssociationManager::new()),
        }
    }

    /// Initialize all native features
    pub async fn initialize<R: Runtime>(&self, app: &AppHandle<R>) -> Result<(), Box<dyn std::error::Error>> {
        // Initialize system tray
        self.system_tray.initialize(app).await?;
        
        // Register file associations
        self.file_associations.register_associations().await?;
        
        info!("All native features initialized successfully");
        Ok(())
    }
}

// Tauri commands for native features
#[tauri::command]
pub async fn show_native_file_dialog(
    app: AppHandle,
    options: FileDialogOptions,
) -> Result<Vec<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    
    let dialog = app.dialog().file();
    let mut builder = dialog.set_title(&options.title);
    
    if let Some(path) = &options.default_path {
        builder = builder.set_directory(path);
    }
    
    for filter in &options.filters {
        builder = builder.add_filter(&filter.name, &filter.extensions);
    }
    
    let result = if options.directory {
        if options.multiple {
            return Err("Multiple directory selection not supported".to_string());
        } else {
            match builder.pick_folder().await {
                Some(path) => vec![path.to_string_lossy().to_string()],
                None => vec![],
            }
        }
    } else if options.multiple {
        match builder.pick_files().await {
            Some(paths) => paths.into_iter()
                .map(|p| p.to_string_lossy().to_string())
                .collect(),
            None => vec![],
        }
    } else {
        match builder.pick_file().await {
            Some(path) => vec![path.to_string_lossy().to_string()],
            None => vec![],
        }
    };
    
    debug!("File dialog result: {:?}", result);
    Ok(result)
}

#[tauri::command]
pub async fn show_save_dialog(
    app: AppHandle,
    title: String,
    default_filename: Option<String>,
    filters: Vec<FileDialogFilter>,
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    
    let dialog = app.dialog().file();
    let mut builder = dialog.set_title(&title);
    
    if let Some(filename) = default_filename {
        builder = builder.set_file_name(&filename);
    }
    
    for filter in &filters {
        builder = builder.add_filter(&filter.name, &filter.extensions);
    }
    
    let result = builder.save_file().await
        .map(|path| path.to_string_lossy().to_string());
    
    debug!("Save dialog result: {:?}", result);
    Ok(result)
}

#[tauri::command]
pub async fn send_native_notification(
    app: AppHandle,
    data: NotificationData,
) -> Result<(), String> {
    use tauri_plugin_notification::NotificationExt;
    
    let mut builder = app
        .notification()
        .builder()
        .title(&data.title)
        .body(&data.body);
    
    if let Some(icon) = &data.icon {
        builder = builder.icon(icon);
    }
    
    // Note: Sound and urgency levels may need platform-specific handling
    // For now, we'll use the basic notification
    
    let notification = builder.build().map_err(|e| e.to_string())?;
    notification.show().map_err(|e| e.to_string())?;
    
    info!("Native notification sent: {}", data.title);
    Ok(())
}

#[tauri::command]
pub async fn handle_drag_drop_event(
    event: DragDropEvent,
    app: AppHandle,
) -> Result<(), String> {
    // Emit drag drop event to frontend
    app.emit("drag-drop", &event)
        .map_err(|e| e.to_string())?;
    
    info!("Drag drop event handled: {} files", event.files.len());
    Ok(())
}

#[tauri::command]
pub async fn update_tray_status(
    state: tauri::State<'_, NativeFeaturesState>,
    process_state: String,
    health_status: String,
) -> Result<(), String> {
    use crate::process_manager::{ProcessState, HealthStatus};
    
    let proc_state = match process_state.as_str() {
        "stopped" => ProcessState::Stopped,
        "starting" => ProcessState::Starting,
        "running" => ProcessState::Running,
        "stopping" => ProcessState::Stopping,
        "crashed" => ProcessState::Crashed,
        "restarting" => ProcessState::Restarting,
        _ => ProcessState::Stopped,
    };
    
    let health = match health_status.as_str() {
        "healthy" => HealthStatus::Healthy,
        "degraded" => HealthStatus::Degraded,
        "unhealthy" => HealthStatus::Unhealthy,
        _ => HealthStatus::Unknown,
    };
    
    state.system_tray.update_server_status(proc_state, health).await
        .map_err(|e| e.to_string())?;
    
    Ok(())
}