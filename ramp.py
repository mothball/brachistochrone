import streamlit as st
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import cumtrapz, quad
import plotly.graph_objects as go


def solve_brachistochrone(x_f, y_i):
    """Solve for r and theta using fsolve."""
    def equations(vars):
        r, theta_f = vars
        eq1 = r * (theta_f - np.sin(theta_f)) - x_f
        eq2 = r * (1 - np.cos(theta_f)) - y_i
        return [eq1, eq2]
    
    r_guess, theta_f_guess = x_f / (2 * np.pi), 1 * np.pi  # Initial guesses
    return fsolve(equations, [r_guess, theta_f_guess])

def brachistochrone_time_and_curve(r, theta_f, y_i, g=9.81, num_points=1000):
    """Generate the Brachistochrone curve and calculate time."""
    theta = np.linspace(0, theta_f, num_points)
    x = r * (theta - np.sin(theta))
    y = y_i - r * (1 - np.cos(theta))

    # Calculate time for each theta
    time = np.zeros_like(theta)
    for i in range(1, len(theta)):
        ds = np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)
        v = np.sqrt(2 * g * (y_i - y[i]))
        dt = ds / v if v > 0 else 0  # Avoid division by zero
        time[i] = time[i - 1] + dt

    return x, y, time

def find_theta_for_x(x, r):
    """Numerically solve for theta given x and r."""
    theta_func = lambda theta: r * (theta - np.sin(theta)) - x
    theta_guess = x / r  # Initial guess for theta based on x
    theta_solution, = fsolve(theta_func, theta_guess)
    return theta_solution

def time_to_reach_x(x, r, y_i, g=9.81):
    """Calculate the time taken to reach a given x along the Brachistochrone curve."""
    # Solve for theta that corresponds to x
    theta_x = find_theta_for_x(x, r)

    # Define the differential arc length and time function
    def dt(theta):
        dx_dtheta = r * (1 - np.sin(theta))
        dy_dtheta = r * np.cos(theta)
        ds = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
        y = y_i - r * (1 - np.cos(theta))
        v = np.sqrt(2 * g * (y_i - y))
        if v < 1e-5:  # Avoid division by very small velocities
            v = 1e-5
        return ds / v

    # Avoid integrating from exactly zero if that's where velocity goes to zero
    start_theta = 1e-6 if theta_x > 1e-6 else theta_x

    # Numerically integrate the time differential from start_theta to theta_x
    if theta_x > start_theta:
        time, _ = quad(dt, start_theta, theta_x)
    else:
        time = 0  # If the angle to reach is less than start_theta, assume negligible time

    return time

def time_function(r, theta, y_i, g):
    """Calculate time along the curve."""
    dx_dtheta = r * (1 - np.sin(theta))
    dy_dtheta = r * np.cos(theta)
    ds = np.sqrt(dx_dtheta**2 + dy_dtheta**2)
    heights = y_i - r * (1 - np.cos(theta))
    v = np.sqrt(2 * g * abs(heights))
    dt = ds / v
    return cumtrapz(dt, theta, initial=0)

def calculate_ramp_angle(x_f, y_i):
    """Calculate the angle of the ramp with respect to the horizontal."""
    return np.arctan(y_i / x_f)

def ramp_acceleration(x_f, y_i, g):
    """Calculate the acceleration down the ramp."""
    theta = calculate_ramp_angle(x_f, y_i)
    return g * np.sin(theta)

def position_on_ramp(t, x_f, y_i, g):
    """Calculate the position along the ramp at time t."""
    a = ramp_acceleration(x_f, y_i, g)
    theta = calculate_ramp_angle(x_f, y_i)
    s = 0.5 * a * t**2  # Distance along the ramp
    x = s * np.cos(theta)  # Horizontal position
    y = y_i - s * np.sin(theta)  # Vertical position, considering decrease from initial height
    return x, y

def ramp_total_time(x_f, y_i, g):
    """Calculate the time for ball to traverse the straight ramp."""
    t_f = np.sqrt(2*(x_f**2+y_i**2)/(g*y_i))
    return t_f

def ramp_time_to_reach_x(x, x_f, y_i, g):
    """Calculate the time to reach a specific x on the straight ramp."""
    theta = np.arctan(y_i / x_f)
    a = g * np.sin(theta)
    s = x / np.cos(theta)  # Calculate the actual path length s traveled along the ramp
    return np.sqrt(2 * s / a)

# Streamlit application layout
st.title('Brachistochrone vs. Straight Ramp')

# Sliders for end points
g = float(st.sidebar.text_input("Gravity (m/s^2)", 9.81))
y_i = st.sidebar.slider("Start Point Y (m)", 1, 100, 22)
x_f = st.sidebar.slider("End Point X (m)", 10, 150, 30)

# Calculate the parameters
r, theta_f = solve_brachistochrone(x_f, y_i)
x, y, t = brachistochrone_time_and_curve(r, theta_f, y_i, g)

# Slider for controlling time within the animation
tf = ramp_total_time(x_f, y_i, g)
time_slider = st.slider("Select Time", 0.0, tf, 0.0, 0.01)

# Find the index closest to the selected time
time_index = np.argmin(np.abs(t - time_slider))

# Ramp calcuations
x_ramp, y_ramp = position_on_ramp(time_slider, x_f, y_i, g)

# Create the plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Brachistochrone'))
fig.add_trace(go.Scatter(x=[x[time_index]], y=[y[time_index]], mode='markers', marker=dict(color='red', size=10), name='Ball'))
fig.add_trace(go.Scatter(x=[0,x_f], y=[y_i,0], mode='lines', name='Ramp'))
fig.add_trace(go.Scatter(x=[x_ramp], y=[y_ramp], mode='markers', marker=dict(color='green', size=10), name='Ramp Ball'))

fig.update_layout(title="Brachistochrone vs Ramp", xaxis_title="X (m)", yaxis_title="Y (m)")
fig.update_xaxes(range=[-1, x_f+1])
fig.update_yaxes(range=[None, y_i+1])
st.plotly_chart(fig, use_container_width=True)

# Display current location of the ball
st.write(f"Brachistochrone Location: X = {x[time_index]:.2f}, Y = {y[time_index]:.2f}")
st.write(f"Ramp Location: X = {x_ramp:.2f}, Y = {y_ramp:.2f}")

# Display table for positions
st.write("Ball Locations:")
# Define the range for x and compute y for each x using the solved r
x_values = np.arange(0, x_f + 1, 5)
y_values = [y_i - r * (1 - np.cos(find_theta_for_x(x, r))) for x in x_values]
y_ramp = [22 - (y_i/x_f)*x for x in x_values]

brachistochrone_times = np.interp(x_values, x, t)
ramp_times = [ramp_time_to_reach_x(x, x_f, y_i, g) for x in x_values]

table_data = {"X (m)": x_values, "Y - Brachistochrone (m)": y_values, "Time - Brachistochrone (s)": brachistochrone_times, "Y - Ramp (m)": y_ramp, "Time - Ramp (s)": ramp_times}
st.table(table_data)