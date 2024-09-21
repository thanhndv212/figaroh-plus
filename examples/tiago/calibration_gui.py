import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess


def select_file(entry):
    filename = filedialog.askopenfilename()
    entry.delete(0, tk.END)
    entry.insert(0, filename)


def run_calibration():
    urdf_path = urdf_entry.get()
    config_path = config_entry.get()
    remove_param = remove_param_entry.get()
    plot_level = plot_level_var.get()
    end_effector = end_effector_entry.get()
    save_calibration = save_var.get()

    cmd = [
        "python",
        "calibration.py",
        "--urdf_path",
        urdf_path,
        "--config_path",
        config_path,
        "--plot_level",
        str(plot_level),
        "--end_effector",
        end_effector,
    ]

    if remove_param:
        cmd.extend(["--remove_param", remove_param])

    if save_calibration:
        cmd.append("--save_calibration")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        messagebox.showinfo("Success", "Calibration completed successfully!")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Calibration failed:\n\n{e.stderr}")


# Create main window
root = tk.Tk()
root.title("TIAGo Robot Calibration")

# URDF file selection
tk.Label(root, text="URDF Path:").grid(row=0, column=0, sticky="e")
urdf_entry = tk.Entry(root, width=50)
urdf_entry.grid(row=0, column=1)
urdf_entry.insert(0, "urdf/tiago_48_schunk.urdf")
tk.Button(
    root, text="Browse", command=lambda: select_file(urdf_entry)
).grid(row=0, column=2)

# Config file selection
tk.Label(root, text="Config Path:").grid(row=1, column=0, sticky="e")
config_entry = tk.Entry(root, width=50)
config_entry.grid(row=1, column=1)
config_entry.insert(0, "config/tiago_config_mocap_vicon.yaml")
tk.Button(
    root, text="Browse", command=lambda: select_file(config_entry)
).grid(row=1, column=2)

# Remove parameter
tk.Label(root, text="Remove Parameter:").grid(row=2, column=0, sticky="e")
remove_param_entry = tk.Entry(root, width=50)
remove_param_entry.grid(row=2, column=1)

# Plot level
tk.Label(root, text="Plot Level:").grid(row=3, column=0, sticky="e")
plot_level_var = tk.IntVar(value=1)
tk.Spinbox(
    root, from_=1, to=5, textvariable=plot_level_var, width=5
).grid(row=3, column=1, sticky="w")

# End effector
tk.Label(root, text="End Effector:").grid(row=4, column=0, sticky="e")
end_effector_entry = tk.Entry(root, width=50)
end_effector_entry.grid(row=4, column=1)
end_effector_entry.insert(0, "schunk")

# Save calibration
save_var = tk.BooleanVar()
tk.Checkbutton(root, text="Save Calibration", variable=save_var).grid(
    row=5, column=1, sticky="w"
)

# Run button
tk.Button(root, text="Run Calibration", command=run_calibration).grid(
    row=6, column=1
)

root.mainloop()