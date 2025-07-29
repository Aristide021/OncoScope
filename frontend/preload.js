const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('api', {
  // App info
  getVersion: () => ipcRenderer.invoke('get-app-version'),
  
  // File operations
  showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
  showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
  
  // Preferences
  getPreference: (key) => ipcRenderer.invoke('get-preference', key),
  setPreference: (key, value) => ipcRenderer.invoke('set-preference', key, value),
  
  // Menu events
  onMenuAction: (callback) => {
    ipcRenderer.on('menu-open-file', callback);
    ipcRenderer.on('menu-save-results', callback);
    ipcRenderer.on('menu-export-pdf', callback);
  },
  
  // Remove listeners
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel);
  }
});