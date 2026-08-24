import matplotlib.pyplot as plt
import matplotlib.tri as tri
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

class SurgeVisualizer:
    def __init__(self, gdf, cmap='RdYlBu_r', projection=ccrs.PlateCarree()):
        """
        Initializes the visualizer with the GeoDataFrame and geographic settings.
        
        Args:
            gdf: GeoDataFrame containing 'Longitude' and 'Latitude'.
            cmap: Colormap for surge elevation or velocity.
            projection: Cartopy projection (defaulting to PlateCarree for lat/lon).
        """
        self.gdf = gdf
        self.lon = gdf.Longitude.values
        self.lat = gdf.Latitude.values
        self.projection = projection
        self.cmap = cmap
        
        # Pre-compute triangulation for the unstructured mesh to save time
        self.triangulation = tri.Triangulation(self.lon, self.lat)

    def _setup_ax(self, ax, extent=None):
        """Standardizes map features for each subplot."""
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.3)
        gl.top_labels = gl.right_labels = False

    def plot_timestep(self, data, ax=None, title=None, vmin=None, vmax=None, extent=None):
        """Plots a single contour map for a specific data array."""
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': self.projection}, figsize=(10, 8))
        
        self._setup_ax(ax, extent)
        
        # Use tricontourf for unstructured ADCIRC/SWAN-style mesh nodes
        contour = ax.tricontourf(self.triangulation, data, levels=20, 
                                 cmap=self.cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        
        if title:
            ax.set_title(title, fontsize=10)
        return contour

    def plot_horizon_comparison(self, ground_truth, predictions, horizons, extent=None):
        """
        Dynamically generates a grid of plots comparing Truth vs Prediction 
        across multiple horizons (the '15 plots' replacement).
        """
        num_horizons = len(horizons)
        fig, axes = plt.subplots(num_horizons, 2, figsize=(12, 4 * num_horizons),
                                 subplot_kw={'projection': self.projection})
        
        # Determine global min/max for consistent color scaling
        vmin = min(ground_truth.min(), predictions.min())
        vmax = max(ground_truth.max(), predictions.max())

        for i, h in enumerate(horizons):
            # Plot Ground Truth (Left Column)
            self.plot_timestep(ground_truth[h], ax=axes[i, 0], 
                              title=f"Truth: T+{h}h", vmin=vmin, vmax=vmax, extent=extent)
            
            # Plot Prediction (Right Column)
            im = self.plot_timestep(predictions[h], ax=axes[i, 1], 
                                   title=f"Pred: T+{h}h", vmin=vmin, vmax=vmax, extent=extent)

        # Add a single colorbar for the whole figure
        fig.subplots_adjust(right=0.85, wspace=0.1, hspace=0.2)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Water Elevation (m)')
        plt.show()
        
viz = SurgeVisualizer(gdf)

# Create the interactive figure
fig = viz.plot_interactive(
    data=y_pred_sample[24], 
    title="Hurricane Surge Prediction (T+24h)",
    color_label="Water Level (m)"
)

# Show in Jupyter or save to file
fig.show()
viz.save_interactive_html(fig, "forecast_T24.html")