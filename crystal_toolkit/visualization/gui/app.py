import gradio as gr
import numpy as np
import plotly.graph_objs as go
import os
import sys
import warnings
warnings.filterwarnings("ignore", message=".*HTTP_422_UNPROCESSABLE_ENTITY.*")


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crystal_toolkit.lattice.lattice import Lattice
from crystal_toolkit.detector.detector import Detector
from crystal_toolkit.detector.detector_config import DetectorConfig
from crystal_toolkit.visualization.composite.hybrid_plotters import KSpace2D, KSpace3D
from crystal_toolkit.visualization.plot_1d.detector_1d import Detector1DPlotter

# ────────────────────────── Helper functions ──────────────────────────


def clean_path(path_str: str) -> str:
    """Strip surrounding single/double quotes from a file path."""
    path_str = path_str.strip()
    while True:
        for q in ('"', "'"):
            if path_str.startswith(q) and path_str.endswith(q):
                path_str = path_str[1:-1].strip()
                break
        else:
            break
    return path_str


def _read_rows(df, ncols: int):
    """Yield non-empty rows from a Gradio Dataframe (list-of-lists)."""
    for row in df:
        if (
            row
            and len(row) >= ncols
            and all(v is not None and str(v).strip() != "" for v in row[:ncols])
        ):
            yield row


def parse_mag_points(df):
    return [[float(r[0]), float(r[1]), float(r[2])] for r in _read_rows(df, 3)]


def parse_phi_theta(df):
    phi_r, theta_r = [], []
    for row in _read_rows(df, 4):
        phi_r.append([float(row[0]), float(row[1])])
        theta_r.append([float(row[2]), float(row[3])])
    return phi_r, theta_r


def parse_q_points(df):
    return [[float(r[0]), float(r[1]), float(r[2])] for r in _read_rows(df, 3)]


def safe_eval_thickness(expr: str, lattice):
    """Evaluate expressions like ``a_star/20`` against lattice data."""
    a = lattice.lattice_data.a_star_par
    b = lattice.lattice_data.b_star_par
    c = lattice.lattice_data.c_star_par
    try:
        return float(
            eval(expr, {"__builtins__": {}}, {"a_star": a, "b_star": b, "c_star": c})
        )
    except Exception:
        try:
            return float(expr)
        except Exception:
            return a / 20


import json
import tempfile
from datetime import datetime


# ──────────────────── Config save / load helpers ────────────────────


def save_config_to_file(
    cif_path_raw, n_h, n_k, n_l, mag_points_df,
    u_h, u_k, u_l, v_h, v_k, v_l, w_h, w_k, w_l,
    incident_energy, psi_min, psi_max, phi_theta_df,
    slice_number, angle_step, plot_type,
    norm_h, norm_k, norm_l, pp_h, pp_k, pp_l, ne_h, ne_k, ne_l,
    thickness_expr,
    q_points_df, width_expr,
    plot_detectors, plot_magnetic_peaks,
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "version": 1,
        "cif_path": clean_path(cif_path_raw),
        "n_h": int(n_h), "n_k": int(n_k), "n_l": int(n_l),
        "mag_points": mag_points_df,
        "u_h": float(u_h), "u_k": float(u_k), "u_l": float(u_l),
        "v_h": float(v_h), "v_k": float(v_k), "v_l": float(v_l),
        "w_h": float(w_h), "w_k": float(w_k), "w_l": float(w_l),
        "incident_energy": float(incident_energy),
        "psi_min": float(psi_min), "psi_max": float(psi_max),
        "phi_theta": phi_theta_df,
        "slice_number": int(slice_number),
        "angle_step": float(angle_step),
        "plot_type": plot_type,
        "norm_h": float(norm_h), "norm_k": float(norm_k), "norm_l": float(norm_l),
        "pp_h": float(pp_h), "pp_k": float(pp_k), "pp_l": float(pp_l),
        "ne_h": float(ne_h), "ne_k": float(ne_k), "ne_l": float(ne_l),
        "thickness": thickness_expr,
        "q_points": q_points_df,
        "width": width_expr,
        "plot_detectors": bool(plot_detectors),
        "plot_magnetic_peaks": bool(plot_magnetic_peaks),
    }
    import os
    tmp_path = os.path.join(tempfile.gettempdir(), f"detector_config_{ts}.json")
    with open(tmp_path, "w") as f:
        json.dump(config, f, indent=2)
    return tmp_path


def load_config_from_file(file):
    fp = file if isinstance(file, str) else getattr(file, "path", str(file))
    with open(fp) as f:
        cfg = json.load(f)
    return (
        gr.update(value=cfg.get("cif_path", "")),
        gr.update(value=cfg.get("n_h", 3)),
        gr.update(value=cfg.get("n_k", 3)),
        gr.update(value=cfg.get("n_l", 5)),
        gr.update(value=cfg.get("mag_points", [[0.5, 0, 0]])),
        gr.update(value=cfg.get("u_h", 0)),
        gr.update(value=cfg.get("u_k", 0)),
        gr.update(value=cfg.get("u_l", 1)),
        gr.update(value=cfg.get("v_h", 1)),
        gr.update(value=cfg.get("v_k", 0)),
        gr.update(value=cfg.get("v_l", 0)),
        gr.update(value=cfg.get("w_h", 0)),
        gr.update(value=cfg.get("w_k", 0)),
        gr.update(value=cfg.get("w_l", 0)),
        gr.update(value=cfg.get("incident_energy", 20)),
        gr.update(value=cfg.get("psi_min", 0)),
        gr.update(value=cfg.get("psi_max", 180)),
        gr.update(value=cfg.get("phi_theta", [[-20, 30, -10, 10], [30, 50, -7, 7]])),
        gr.update(value=cfg.get("slice_number", 10)),
        gr.update(value=cfg.get("angle_step", 2)),
        gr.update(value=cfg.get("plot_type", "3D K-space")),
        gr.update(value=cfg.get("norm_h", 0)),
        gr.update(value=cfg.get("norm_k", 0)),
        gr.update(value=cfg.get("norm_l", 1)),
        gr.update(value=cfg.get("pp_h", 0)),
        gr.update(value=cfg.get("pp_k", 0)),
        gr.update(value=cfg.get("pp_l", 0)),
        gr.update(value=cfg.get("ne_h", 1)),
        gr.update(value=cfg.get("ne_k", 0)),
        gr.update(value=cfg.get("ne_l", 0)),
        gr.update(value=cfg.get("thickness", "a_star/20")),
        gr.update(value=cfg.get("q_points", [[0, 0, 0], [-1, 0, 2], [-1, 0, -1]])),
        gr.update(value=cfg.get("width", "a_star/20")),
        gr.update(value=cfg.get("plot_detectors", True)),
        gr.update(value=cfg.get("plot_magnetic_peaks", True)),
    )


def save_plot_html(fig):
    if fig is None:
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    import os
    tmp_path = os.path.join(tempfile.gettempdir(), f"detector_plot_{ts}.html")
    fig.write_html(tmp_path)
    return tmp_path


# ─────────────────────── File-upload callback ────────────────────────


def on_file_upload(file):
    if file is None:
        return gr.update()
    path = file if isinstance(file, str) else getattr(file, "path", str(file))
    return gr.update(value=clean_path(path))


# ──────────────────────── Plot generation ──────────────────────────


def generate_plot(
    cif_path_raw,
    n_h,
    n_k,
    n_l,
    mag_points_df,
    u_h,
    u_k,
    u_l,
    v_h,
    v_k,
    v_l,
    w_h,
    w_k,
    w_l,
    incident_energy,
    psi_min,
    psi_max,
    phi_theta_df,
    slice_number,
    angle_step,
    plot_type,
    norm_h,
    norm_k,
    norm_l,
    pp_h,
    pp_k,
    pp_l,
    ne_h,
    ne_k,
    ne_l,
    thickness_expr,
    q_points_df,
    width_expr,
    plot_detectors,
    plot_magnetic_peaks,
):
    try:
        # --- CIF ---
        cif_path = clean_path(cif_path_raw)
        lattice = Lattice.from_cif(cif_path, [int(n_h), int(n_k), int(n_l)])

        # --- Magnetic peaks ---
        mag_pts = parse_mag_points(mag_points_df)
        if mag_pts:
            lattice.set_magnetic_points(mag_pts)

        # --- Vectors ---
        u = lattice.get_hkl_vector(float(u_h), float(u_k), float(u_l))
        v = lattice.get_hkl_vector(float(v_h), float(v_k), float(v_l))
        if float(w_h) == 0 and float(w_k) == 0 and float(w_l) == 0:
            w = None
        else:
            w = lattice.get_hkl_vector(float(w_h), float(w_k), float(w_l))

        # --- Detector config ---
        phi_ranges, theta_ranges = parse_phi_theta(phi_theta_df)
        detector_config = DetectorConfig(
            incident_energy=float(incident_energy),
            detector_u=u,
            detector_v=v,
            detector_w=w,
            theta_ranges_direct=theta_ranges,
            phi_ranges=phi_ranges,
            psi_range=[float(psi_min), float(psi_max)],
        )
        detector = Detector(
            detector_config,
            slice_number=int(slice_number),
            angle_step=float(angle_step),
        )

        # --- Plot by type ---
        if plot_type == "3D K-space":
            fig = KSpace3D(lattice, detector).plot(
                is_plot_detectors=plot_detectors,
                is_plot_magnetic_peaks=plot_magnetic_peaks,
            )
        elif plot_type == "2D K-space":
            norm = lattice.get_hkl_vector(float(norm_h), float(norm_k), float(norm_l))
            plane_point = lattice.get_hkl_vector(float(pp_h), float(pp_k), float(pp_l))
            new_ex = lattice.get_hkl_vector(float(ne_h), float(ne_k), float(ne_l))
            thickness = safe_eval_thickness(thickness_expr, lattice)
            fig = KSpace2D(
                lattice, norm, plane_point, thickness, new_ex, detector
            ).plot(
                is_plot_detectors=plot_detectors,
                is_plot_magnetic_peaks=plot_magnetic_peaks,
            )
        else:  # 1D Q-E
            q_hkl_list = parse_q_points(q_points_df)
            q_points = [
                lattice.get_hkl_vector(float(q[0]), float(q[1]), float(q[2]))
                for q in q_hkl_list
            ]
            width = safe_eval_thickness(width_expr, lattice)
            fig = Detector1DPlotter(
                detector,
                q_points,
                width,
                lattice.lattice_data.conv_reciprocal_matrix,
            ).plot()

        return fig, fig

    except Exception as e:
        import traceback

        traceback.print_exc()
        import plotly.graph_objs as go

        fig = go.Figure()
        fig.add_annotation(
            text=f"<b>Error:</b> {str(e)}<br><span style='font-size:10px'>See terminal.</span>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="red", size=14),
        )
        fig.update_layout(
            xaxis=dict(visible=False, range=[0, 1]),
            yaxis=dict(visible=False, range=[0, 1]),
            template="plotly_white",
        )
        return fig, fig


# Initial empty figure (so the Plot component renders on first load)
_init_fig = go.Figure()
_init_fig.update_layout(
    template="plotly_white",
    xaxis=dict(visible=False, range=[0, 1]),
    yaxis=dict(visible=False, range=[0, 1]),
)

# ──────────────────────────── UI Layout ─────────────────────────────

PLOT_TYPES = ["3D K-space", "2D K-space", "1D Q-E"]

with gr.Blocks(title="Neutron Scattering Coverage Simulation", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        "# 🔬 Neutron Scattering Coverage Simulation\n"
        "Crystal, detector and plot settings are shared across all plot types."
    )

    with gr.Row():
        toggle_sidebar = gr.Button(
            "◀  Sidebar", size="sm", scale=0, min_width=60,
        )
 
    with gr.Row():
        # ── Left column: inputs ──
        with gr.Column(scale=1) as left_col:

            # ══════════ Crystal / Detector ══════════
            with gr.Group():
                gr.Markdown("### Crystal / Detector Settings")

                with gr.Row():
                    cif_path = gr.Textbox(
                        label="CIF File Path",
                        placeholder="/path/to/your/file.cif",
                        value="path/to/your.cif",
                        scale=4,
                    )
                    cif_upload = gr.UploadButton(
                        "📁 Browse",
                        file_types=[".cif"],
                        file_count="single",
                        scale=1,
                        min_width=80,
                    )

                with gr.Row():
                    n_h = gr.Number(label="Reciprocal N H", value=3, precision=0)
                    n_k = gr.Number(label="K", value=3, precision=0)
                    n_l = gr.Number(label="L", value=5, precision=0)

                mag_points_df = gr.Dataframe(
                    headers=["H", "K", "L"],
                    label="Magnetic Bragg Peaks (each row = one propagation vector)",
                    row_count=(2, "dynamic"),
                    col_count=(3, "fixed"),
                    type="array",
                    value=[[0.5, 0, 0]],
                    interactive=True,
                )

                gr.Markdown("#### Detector Vectors (HKL)")
                with gr.Row():
                    u_h = gr.Number(label="u H", value=0)
                    u_k = gr.Number(label="u K", value=0)
                    u_l = gr.Number(label="u L", value=1)
                with gr.Row():
                    v_h = gr.Number(label="v H", value=1)
                    v_k = gr.Number(label="v K", value=0)
                    v_l = gr.Number(label="v L", value=0)
                with gr.Row():
                    w_h = gr.Number(label="w H (0=auto)", value=0)
                    w_k = gr.Number(label="w K (0=auto)", value=0)
                    w_l = gr.Number(label="w L (0=auto)", value=0)

                with gr.Row():
                    incident_energy = gr.Number(label="Incident Energy (meV)", value=20)
                    slice_number = gr.Number(
                        label="Slice Number", value=10, precision=0
                    )
                    angle_step = gr.Number(label="Angle Step (°)", value=2)

                with gr.Row():
                    psi_min = gr.Number(label="Psi Min (°)", value=0)
                    psi_max = gr.Number(label="Psi Max (°)", value=180)

                # Phi / Theta in one table — rows map 1-to-1
                phi_theta_df = gr.Dataframe(
                    headers=[
                        "Phi Min (°)",
                        "Phi Max (°)",
                        "Theta Min (°)",
                        "Theta Max (°)",
                    ],
                    label="Phi / Theta Ranges — each row is one (phi, theta) pair",
                    row_count=(2, "dynamic"),
                    col_count=(4, "fixed"),
                    type="array",
                    value=[[-20, 30, -10, 10], [30, 50, -7, 7]],
                    interactive=True,
                )

            # ══════════ Plot controls ══════════
            with gr.Group():
                gr.Markdown("### Plot Settings")
                plot_type = gr.Radio(
                    choices=PLOT_TYPES,
                    label="Plot Type",
                    value="3D K-space",
                )
                with gr.Row():
                    with gr.Column(visible=True) as plot_toggles:
                        with gr.Row():
                            plot_detectors = gr.Checkbox(
                                label="Draw Detector Coverage", value=True
                            )
                            plot_magnetic_peaks = gr.Checkbox(
                                label="Draw Magnetic Peaks", value=True
                            )

            # ══════════ 2D params ══════════
            with gr.Column(visible=False) as params_2d:
                with gr.Group():
                    gr.Markdown("#### 2D Settings")
                    with gr.Row():
                        norm_h = gr.Number(label="Norm H", value=0)
                        norm_k = gr.Number(label="Norm K", value=0)
                        norm_l = gr.Number(label="Norm L", value=1)
                    with gr.Row():
                        pp_h = gr.Number(label="Plane Point H", value=0)
                        pp_k = gr.Number(label="Plane Point K", value=0)
                        pp_l = gr.Number(label="Plane Point L", value=0)
                    with gr.Row():
                        ne_h = gr.Number(label="New Ex H", value=1)
                        ne_k = gr.Number(label="New Ex K", value=0)
                        ne_l = gr.Number(label="New Ex L", value=0)
                    thickness = gr.Textbox(
                        label="Thickness (a_star/n)", value="a_star/20"
                    )

            # ══════════ 1D params ══════════
            with gr.Column(visible=False) as params_1d:
                with gr.Group():
                    gr.Markdown("#### 1D Settings")
                    q_points_df = gr.Dataframe(
                        headers=["H", "K", "L"],
                        label="Q Points (HKL) — one point per row",
                        row_count=(3, "dynamic"),
                        col_count=(3, "fixed"),
                        type="array",
                        value=[[0, 0, 0], [-1, 0, 2], [-1, 0, -1]],
                        interactive=True,
                    )
                    width = gr.Textbox(label="Width (a_star/n)", value="a_star/20")

            # ══════════ Generate ══════════
            generate_btn = gr.Button("Generate Plot", variant="primary", size="lg")
            with gr.Row():
                save_config_btn = gr.DownloadButton("💾 Save Config")
                load_config_btn = gr.UploadButton(
                    "📂 Load Config", file_types=[".json"], file_count="single"
                )
                save_plot_btn = gr.DownloadButton("💾 Save Plot (HTML)")

        # ── Right column: output ──
        with gr.Column(scale=2) as plot_col:
            plot_output = gr.Plot(label="Result", show_label=True, value=_init_fig)
            last_fig_state = gr.State(None)

    # ──────────────────── Event wiring ────────────────────

    # Browse → update textbox
    cif_upload.upload(on_file_upload, inputs=cif_upload, outputs=cif_path)

    # Plot type → show/hide sections
    def _on_type_change(t):
        show_toggles = t != "1D Q-E"
        return (
            gr.update(visible=(t == "2D K-space")),
            gr.update(visible=(t == "1D Q-E")),
            gr.update(visible=show_toggles),
        )

    plot_type.change(_on_type_change, plot_type, [params_2d, params_1d, plot_toggles])

    # Detector toggle
    def _on_detector_toggle(v):
        s = "" if v else " [⚠️ not plotted]"
        return (
            gr.update(
                label=f"Phi / Theta Ranges — each row is one (phi, theta) pair{s}"
            ),
            gr.update(label=f"Psi Min (°){s}"),
            gr.update(label=f"Psi Max (°){s}"),
            gr.update(label=f"Slice Number{s}"),
            gr.update(label=f"Angle Step (°){s}"),
        )

    plot_detectors.change(
        _on_detector_toggle,
        plot_detectors,
        [phi_theta_df, psi_min, psi_max, slice_number, angle_step],
    )

    # Magnetic toggle
    plot_magnetic_peaks.change(
        lambda v: gr.update(
            label=f"Magnetic Bragg Peaks{' [⚠️ not plotted]' if not v else ''}"
        ),
        plot_magnetic_peaks,
        mag_points_df,
    )

    # Generate
    generate_btn.click(
        fn=generate_plot,
        inputs=[
            cif_path,
            n_h,
            n_k,
            n_l,
            mag_points_df,
            u_h,
            u_k,
            u_l,
            v_h,
            v_k,
            v_l,
            w_h,
            w_k,
            w_l,
            incident_energy,
            psi_min,
            psi_max,
            phi_theta_df,
            slice_number,
            angle_step,
            plot_type,
            norm_h,
            norm_k,
            norm_l,
            pp_h,
            pp_k,
            pp_l,
            ne_h,
            ne_k,
            ne_l,
            thickness,
            q_points_df,
            width,
            plot_detectors,
            plot_magnetic_peaks,
        ],
        outputs=[plot_output, last_fig_state],
    )

    # Save / Load config
    save_config_btn.click(
        fn=save_config_to_file,
        inputs=[
            cif_path, n_h, n_k, n_l, mag_points_df,
            u_h, u_k, u_l, v_h, v_k, v_l, w_h, w_k, w_l,
            incident_energy, psi_min, psi_max, phi_theta_df,
            slice_number, angle_step, plot_type,
            norm_h, norm_k, norm_l, pp_h, pp_k, pp_l, ne_h, ne_k, ne_l,
            thickness, q_points_df, width,
            plot_detectors, plot_magnetic_peaks,
        ],
        outputs=save_config_btn,
    )

    load_config_btn.upload(
        fn=load_config_from_file,
        inputs=load_config_btn,
        outputs=[
            cif_path, n_h, n_k, n_l, mag_points_df,
            u_h, u_k, u_l, v_h, v_k, v_l, w_h, w_k, w_l,
            incident_energy, psi_min, psi_max, phi_theta_df,
            slice_number, angle_step, plot_type,
            norm_h, norm_k, norm_l, pp_h, pp_k, pp_l, ne_h, ne_k, ne_l,
            thickness, q_points_df, width,
            plot_detectors, plot_magnetic_peaks,
        ],
    )

    save_plot_btn.click(
        fn=save_plot_html,
        inputs=last_fig_state,

        outputs=save_plot_btn,
    )

    # Sidebar toggle
    sidebar_visible = gr.State(value=True)
 
    def _on_toggle_sidebar(show, fig):
        new_h = 700 if show else 400
        if fig is not None:
            fig.update_layout(height=new_h, width=None, autosize=True)
        return (
            gr.update(value="▶  Sidebar"),
            gr.update(visible=False, scale=0),
            gr.update(scale=1),
            gr.update(value=fig),
            False,
        ) if show else (
            gr.update(value="◀  Sidebar"),
            gr.update(visible=True, scale=1),
            gr.update(scale=2),
            gr.update(value=fig),
            True,
        )

    toggle_sidebar.click(
        fn=_on_toggle_sidebar,
        inputs=[sidebar_visible, last_fig_state],
        outputs=[toggle_sidebar, left_col, plot_col, plot_output, sidebar_visible],
    )
 
# ───────────────────────────── Main ─────────────────────────────


def _find_free_port(start=7860, max_attempts=20):
    """Return the first available port starting from *start*."""
    import socket
    for port in range(start, start + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start  # fallback


def main():
    demo.launch(
        server_name="127.0.0.1",
        server_port=_find_free_port(),
        inbrowser=True,
    )


if __name__ == "__main__":
    main()
