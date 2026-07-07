# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------

def _try_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ImportError:
        return None, None


def _add_geo(ax, lon, lat, ccrs, cfeature):
    if ccrs is not None:
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle="--")
        ax.gridlines(draw_labels=False, linewidth=0.4, alpha=0.5, linestyle=":")
        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
