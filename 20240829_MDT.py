import numpy as np
from scipy.integrate import solve_ivp
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime

# Fixed parameter values
params = {
    'M_S_CO2': 0.85 * 5.2 - 0.2 * 5.2,
    'M_B_CO2': 0.2 * 5.2,
    'V_Btis_O2': 1000,
    'V_D': 0.181,
    'f_V_cap': 0.01,
    'w': 83.2,
    'M_O2': 5.2,
    'M_CO2': 0.85 * 5.2,
    'V_CO2': 15000,
    'V_O2': 6000,
    'M_B_O2': -0.2 * 5.2,
    'M_S_O2': -5.2 - (-0.2 * 5.2),
    'V_Btis_CO2': 900,
    'V_Stis_CO2': 15000 - 900,
    'V_Stis_O2': 6000 - 1000,
    'V_Bcap_CO2': 0.01 * 900,
    'V_Scap_CO2': 0.01 * (15000 - 900),
    'V_Bcap_O2': 0.01 * 1000,
    'V_Scap_O2': 0.01 * (6000 - 1000),
    'K_O2': 0.200,
    'k_O2': 0.046,
    'K_CO2': 0.0065,
    'k_CO2': 0.244,
    'K_O2_tau': 0.0025,
    'D_T_CO2': 9 / 60 * 83.2 / 0.0065,
    'D_T_O2': 9 / 60 * 83.2 / 0.0025,
}

# Initial conditions
initial_conditions = {
    'p_d1_CO2': 5,
    'p_d1_O2': 159,
    'p_d2_CO2': 6,
    'p_d2_O2': 158,
    'p_d3_CO2': 7,
    'p_d3_O2': 157,
    'p_a_CO2': 40,
    'p_a_O2': 100,
    'c_Stis_CO2': 0.543,
    'c_Stis_O2': 0.128,
    'c_Btis_CO2': 0.569,
    'c_Btis_O2': 0.112,
    'c_Scap_CO2': 0.543 - params['M_S_CO2'] / params['D_T_CO2'],
    'c_Scap_O2': 0.128 + params['M_S_O2'] / params['D_T_O2'],
    'c_Bcap_CO2': 0.569 - params['M_B_CO2'] / params['D_T_CO2'],
    'c_Bcap_O2': 0.112 + params['M_B_O2'] / params['D_T_O2'],
}

# Bloodflows
bloodflows = {
    'CO': 84.5,  # Cardiac output [ml/sec]
    'q_p': 84.5,
    'sh': 0.02,
    'q_Bv': 0.2 * 84.5,
    'q_S': 0.8 * 84.5,
}

# Constants and State-Space Matrices setup
cardio_constants = {
    'C_l': 0.00127,  # l/cmH2O
    'UV_l': 34.4 / 1000,  # ml
    'R_ml': 1.021 * 1.5,  # cmH2O · s · l^(-1)
    'C_tr': 0.00238,  # l/cmH2O
    'UV_tr': 6.63 / 1000,  # ml
    'R_lt': 0.3369,  # cmH2O · s · l^(-1)
    'C_b': 0.0131,  # l/cmH2O
    'UV_b': 18.7 / 1000,  # ml
    'R_tb': 0.3063,  # cmH2O · s · l^(-1)
    'C_A': 0.2,  # l/cmH2O
    'UV_A': 1.263,  # [MODEL]
    'R_bA': 0.0817,  # cmH2O · s · l^(-1)
}

# Parameters for the gas exchange model
gas_exchange_params = {
    'FI_O2': 0.21,
    'FI_CO2': 0.0004,
    'FA_O2': 104 / 713,
    'FA_CO2': 40 / 713,
    'V_D': 0.181,
    'V_A': 2.3,
    'c_S_O2': initial_conditions['c_Scap_O2'],
    'c_S_CO2': initial_conditions['c_Scap_CO2'],
    'c_B_O2': initial_conditions['c_Bcap_O2'],
    'c_B_CO2': initial_conditions['c_Bcap_CO2'],
    'D_S_CO2': params['D_T_CO2'],
    'D_S_O2': params['D_T_O2'],
    'D_B_CO2': params['D_T_CO2'],
    'D_B_O2': params['D_T_O2'],
}

# Derived parameters for gas exchange
gas_exchange_params.update({
    'c_v_CO2': (initial_conditions['c_Stis_CO2'] * bloodflows['q_S'] + initial_conditions['c_Btis_CO2'] * bloodflows['q_Bv']) / (bloodflows['q_S'] + bloodflows['q_Bv']),
    'c_v_O2': (initial_conditions['c_Stis_O2'] * bloodflows['q_S'] + initial_conditions['c_Btis_O2'] * bloodflows['q_Bv']) / (bloodflows['q_S'] + bloodflows['q_Bv']),
    'c_a_CO2': params['K_CO2'] * initial_conditions['p_a_CO2'] + params['k_CO2'],
    'c_a_O2': params['K_O2'] * np.power((1 - np.exp(-params['k_O2'] * min(initial_conditions['p_a_O2'], 700))), 2)
})

# Respiratory control system
respiratory_control_params = {
    'PaCO2_n': 40,
    'Gp_A': -6,
    'Gp_f': 0.8735,
    'Gc_A': -2,
    'Gc_f': 0.9,
    'tau_p_A': 83,
    'tau_p_f': 147.78,
    'tau_c_A': 105,
    'tau_c_f': 400,
    'RR_0': 12,
    'Pmus_0': -5,
    'Delta_RR_c': 0,
    'Delta_RR_p': 0,
    'Delta_Pmus_c': 0,
    'Delta_Pmus_p': 0,
    'f_acp_n': 3.7,
}

cardio_control_params = {
    'ABP_n': 85,
    'HR_n': 70,
    'R_n': 1,
    'UV_n': 1,
    'R_c': 1,
    'UV_c': 1,
    'Gc_hr': .9,
    'Gc_r': 0.05,
    'Gc_uv': 0.05,
    'tau_hr': 105,
    'tau_r': 205,
    'tau_uv': 205,
}

# Other constants
misc_constants = {
    'MV': 0,  # default to spontaneous breathing
    'P_vent': 0,
    'Hgb': 15,  # g/dL
    'HR': 70,  # Heart rate
    'RR': 12,  # Respiratory rate
    'Pintra_t0': -4,  # Initial intra-thoracic pressure
    'RRP': 60 / 12,  # Respiratory rate period
    'TBV': 5000,  # Total blood volume
    'T': 0.01,  # Sample frequency
    'tmin': 0.0,  # Start time
    'tmax': 60.0,  # Simulation duration
    'ncc': 1,  # Cardiac cycle counter
    'N': round((60.0 - 0.0) / 0.01) + 1,  # Number of iterations
    't_span': (0.0, 60.0),
    't_eval': np.arange(0.0, 60.0 + 0.01, 0.01),
}

# Initialize arrays to store model parameters
elastance = np.zeros((2, 11))
resistance = np.zeros(11)
uvolume = np.zeros(11)

# Initialize model parameters
elastance[:, 0] = [1.43, np.nan]  # Intra-thoracic arteries
elastance[:, 1] = [0.6, np.nan]  # Extra-thoracic arteries
elastance[:, 2] = [0.0169, np.nan]  # Extra-thoracic veins
elastance[:, 3] = [0.0182, np.nan]  # Intra-thoracic veins
elastance[:, 4] = [0.05, 0.15]  # Right atrium (min, max)
elastance[:, 5] = [0.057, 0.49]  # Right ventricle (min, max)
elastance[:, 6] = [0.233, np.nan]  # Pulmonary arteries
elastance[:, 7] = [0.0455, np.nan]  # Pulmonary veins
elastance[:, 8] = [0.12, 0.28]  # Left atrium (min, max)
elastance[:, 9] = [0.09, 4]  # Left ventricle (min, max)

resistance = np.array([
    0.06,  # Intra-thoracic arteries
    0.85,  # Extra-thoracic arteries
    0.09,  # Extra-thoracic veins
    0.003,  # Intra-thoracic veins
    0.003,  # Right atrium
    0.003,  # Right ventricle
    0.11,  # Pulmonary arteries
    0.003,  # Pulmonary veins
    0.003,  # Left atrium
    0.008,  # Left ventricle
])

uvolume = np.array([
    140,  # Intra-thoracic arteries
    370,  # Extra-thoracic arteries
    1000,  # Extra-thoracic veins
    1190,  # Intra-thoracic veins
    14,  # Right atrium
    26,  # Right ventricle
    50,  # Pulmonary arteries
    350,  # Pulmonary veins
    11,  # Left atrium
    20,  # Left ventricle
])

# Lung model equations
# Mechanical system parameters
C_cw = 0.2445  # l/cmH2O

A_mechanical = np.array([
    [-1 / (cardio_constants['C_l'] * cardio_constants['R_ml']) - 1 / (cardio_constants['R_lt'] * cardio_constants['C_l']), 1 / (cardio_constants['R_lt'] * cardio_constants['C_l']), 0, 0, 0],
    [1 / (cardio_constants['R_lt'] * cardio_constants['C_tr']), -1 / (cardio_constants['C_tr'] * cardio_constants['R_lt']) - 1 / (cardio_constants['R_tb'] * cardio_constants['C_tr']), 1 / (cardio_constants['R_tb'] * cardio_constants['C_tr']), 0, 0],
    [0, 1 / (cardio_constants['R_tb'] * cardio_constants['C_b']), -1 / (cardio_constants['C_b'] * cardio_constants['R_tb']) - 1 / (cardio_constants['R_bA'] * cardio_constants['C_b']), 1 / (cardio_constants['R_bA'] * cardio_constants['C_b']), 0],
    [0, 0, 1 / (cardio_constants['R_bA'] * cardio_constants['C_A']), -1 / (cardio_constants['C_A'] * cardio_constants['R_bA']), 0],
    [1 / (cardio_constants['R_lt'] * C_cw), -1 / (C_cw * cardio_constants['R_lt']), 0, 0, 0]
])

B_mechanical = np.array([
    [1 / (cardio_constants['R_ml'] * cardio_constants['C_l']), 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Initialize blood volumes based on unstressed volumes
V = misc_constants['TBV'] * (uvolume / np.sum(uvolume))

import numpy as np

import numpy as np

def generate_sinus_ecg_segment(t, heart_rate):
    # Calculate the duration of one cardiac cycle based on the heart rate
    cycle_duration = 60.0 / heart_rate
    segment_duration = t[-1] - t[0]

    # Proportional durations of the ECG components based on cycle duration
    p_wave_duration = 0.1 * cycle_duration
    p_wave_amplitude = 0.2
    qrs_duration = 0.1 * cycle_duration
    qrs_amplitude = 1.5
    t_wave_duration = 0.2 * cycle_duration
    t_wave_amplitude = 0.5

    # Initialize the ECG segment with zeros
    ecg_wave = np.zeros_like(t)

    # Calculate the start times within the cycle for P wave, QRS complex, and T wave
    p_wave_start = 0.1 * cycle_duration
    qrs_start = 0.3 * cycle_duration
    t_wave_start = 0.5 * cycle_duration

    # Modulo the time with the cycle duration to get the position within the cycle
    cycle_position = t % cycle_duration

    # P wave
    mask_p = (cycle_position >= p_wave_start) & (cycle_position < p_wave_start + p_wave_duration)
    ecg_wave[mask_p] = p_wave_amplitude * np.sin(np.pi * (cycle_position[mask_p] - p_wave_start) / p_wave_duration)

    # QRS complex
    mask_qrs = (cycle_position >= qrs_start) & (cycle_position < qrs_start + qrs_duration)
    ecg_wave[mask_qrs] = qrs_amplitude * np.sin(np.pi * (cycle_position[mask_qrs] - qrs_start) / qrs_duration)

    # T wave
    mask_t = (cycle_position >= t_wave_start) & (cycle_position < t_wave_start + t_wave_duration)
    ecg_wave[mask_t] = t_wave_amplitude * np.sin(np.pi * (cycle_position[mask_t] - t_wave_start) / t_wave_duration)

    # Add noise
    noise = np.random.normal(0, 0.02, t.shape)
    ecg_wave += noise

    return ecg_wave


def generate_vt_ecg_segment(t, heart_rate):
    # Calculate the duration of one cardiac cycle based on the heart rate
    heart_rate = 110
    cycle_duration = 60.0 / heart_rate
    segment_duration = t[-1] - t[0]

    # VT is characterized by wide QRS complexes and absent P and T waves
    qrs_duration = 0.2 * cycle_duration  # Wider QRS complex
    qrs_amplitude = 1.0  # Lower amplitude compared to normal QRS

    # Initialize the ECG segment with zeros
    ecg_wave = np.zeros_like(t)

    # Calculate the start time within the cycle for QRS complex
    qrs_start = 0.2 * cycle_duration

    # Modulo the time with the cycle duration to get the position within the cycle
    cycle_position = t % cycle_duration

    # QRS complex (simulating ventricular tachycardia morphology)
    mask_qrs = (cycle_position >= qrs_start) & (cycle_position < qrs_start + qrs_duration)
    ecg_wave[mask_qrs] = qrs_amplitude * np.sign(np.sin(np.pi * (cycle_position[mask_qrs] - qrs_start) / qrs_duration))

    # Add irregularity to the waveform to simulate VT variability
    irregularity = 0.1 * np.sin(2 * np.pi * cycle_position / (cycle_duration * 2))
    ecg_wave += irregularity

    # Add noise
    noise = np.random.normal(0, 0.05, t.shape)
    ecg_wave += noise

    return ecg_wave


def update_heart_period(HR):
    """Calculate heart period parameters."""
    HP = 60 / HR
    Tas = 0.03 + 0.09 * HP
    Tav = 0.01
    Tvs = 0.16 + 0.2 * HP
    return HR, HP, Tas, Tav, Tvs

def calculate_respiratory_curve(Pintra_t0, RRP, t):
    """Calculate the respiratory curve."""
    return Pintra_t0 + (1 + np.cos(2 * np.pi / RRP * t))

def calculate_elastances(t, HR, Tas, Tav, Tvs, T, elastance):
    """Calculate heart elastances."""
    HP = 60/HR

    ncc = (t % HP) / T

    if ncc <= round(Tas / T):
        aaf = np.sin(np.pi * ncc / (Tas / T))
    else:
        aaf = 0

    ela = elastance[0, 8] + (elastance[1, 8] - elastance[0, 8]) * aaf
    era = elastance[0, 4] + (elastance[1, 4] - elastance[0, 4]) * aaf

    if ncc <= round((Tas + Tav) / T):
        vaf = 0
    elif ncc <= round((Tas + Tav + Tvs) / T):
        vaf = np.sin(np.pi * (ncc - (Tas + Tav) / T) / (Tvs / T))
    else:
        vaf = 0

    elv = elastance[0, 9] + (elastance[1, 9] - elastance[0, 9]) * vaf
    erv = elastance[0, 5] + (elastance[1, 5] - elastance[0, 5]) * vaf

    return ela, era, elv, erv

# Initialize heart period parameters
HR, HP, Tas, Tav, Tvs = update_heart_period(misc_constants['HR'])

def get_inputs(t, HR):
    """Get inputs for the cardiovascular system."""
    HR, HP, Tas, Tav, Tvs = update_heart_period(HR)
    ela, era, elv, erv = calculate_elastances(t, HR,Tas, Tav, Tvs, misc_constants['T'], elastance)
    return np.array([ela, elv, era, erv])

# Initialize arrays to store the pressures and flows over time
P_store = np.zeros((10, len(misc_constants['t_eval'])))
F_store = np.zeros((10, len(misc_constants['t_eval'])))
HR_store = np.zeros(len(misc_constants['t_eval']))

# Define the combined system of differential equations
def extended_state_space_equations(t, x):
    global HR
    # Split x into cardiovascular and lung model components
    V = x[:10]  # Volumes for cardiovascular model
    mechanical_states = x[10:15]  # Assuming the next 5 states are mechanical
    FD_O2, FD_CO2, p_a_CO2, p_a_O2 = x[15], x[16], x[17], x[18]
    c_Stis_CO2, c_Scap_CO2, c_Stis_O2, c_Scap_O2 = x[19:23]
    Delta_RR_c = x[23]
    Delta_Pmus_c = x[24]
    Pmus = x[25]
    Delta_HR_c = x[26]
    Delta_R_c = x[27]
    Delta_UV_c = x[28]

    # Inputs for cardiovascular model


    inputs = get_inputs(t, HR)
    ela, elv, era, erv = inputs

    HR = cardio_control_params['HR_n'] - Delta_HR_c
    R_c = cardio_control_params['R_n'] - Delta_R_c
    UV_c = cardio_control_params['UV_n'] + Delta_UV_c

    if misc_constants['MV'] == 0:
        P_ao = 0
        RR = respiratory_control_params['RR_0'] + Delta_RR_c
        Pmus_min = respiratory_control_params['Pmus_0'] + Delta_Pmus_c
        driver = input_function(t, RR, Pmus_min)
        Pmus_dt = driver[1]
        FI_O2 = gas_exchange_params['FI_O2']
        FI_CO2 = gas_exchange_params['FI_CO2']
    else:
        P_ao = ventilator_pressure(t)
        driver = np.array([P_ao, 0])
        RR = 12
        FI_O2 = gas_exchange_params['FI_O2']
        FI_CO2 = gas_exchange_params['FI_CO2']
        Pmus_dt = 0

    # Calculate the pressures for cardiovascular model
    P = np.zeros(10)
    P[0] = elastance[0, 0] * (V[0] - uvolume[0]) + x[14]
    P[1] = elastance[0, 1] * (V[1] - uvolume[1])
    P[2] = elastance[0, 2] * (V[2] - uvolume[2] * UV_c)
    P[3] = elastance[0, 3] * (V[3] - uvolume[3] * UV_c) + x[14]
    P[4] = era * (V[4] - uvolume[4]) + x[14]
    P[5] = erv * (V[5] - uvolume[5]) + x[14]
    P[6] = elastance[0, 6] * (V[6] - uvolume[6]) + x[14]
    P[7] = elastance[0, 7] * (V[7] - uvolume[7]) + x[14]
    P[8] = ela * (V[8] - uvolume[8]) + x[14]
    P[9] = elv * (V[9] - uvolume[9]) + x[14]

    # Calculate the flows for cardiovascular model
    F = np.zeros(10)
    F[0] = (P[0] - P[1]) / (resistance[0] * R_c)
    F[1] = (P[1] - P[2]) / (resistance[1] * R_c)
    F[2] = (P[2] - P[3]) / (resistance[2] * R_c)
    F[3] = (P[3] - P[4]) / resistance[3] if P[3] - P[4] > 0 else (P[3] - P[4]) / (10 * resistance[3])
    F[4] = (P[4] - P[5]) / resistance[4] if P[4] - P[5] > 0 else 0
    F[5] = (P[5] - P[6]) / resistance[5] if P[5] - P[6] > 0 else 0
    F[6] = (P[6] - P[7]) / resistance[6]
    F[7] = (P[7] - P[8]) / resistance[7] if P[7] - P[8] > 0 else (P[7] - P[8]) / (10 * resistance[7])
    F[8] = (P[8] - P[9]) / resistance[8] if P[8] - P[9] > 0 else 0
    F[9] = (P[9] - P[0]) / resistance[9] if P[9] - P[0] > 0 else 0

    # Store pressures and flows in the global arrays
    idx = np.searchsorted(misc_constants['t_eval'], t)
    if 0 <= idx < len(misc_constants['t_eval']):
        P_store[:, idx] = P
        F_store[:, idx] = F
        HR_store[idx] = HR

    # Calculate the derivatives of volumes for cardiovascular model
    dVdt = np.zeros(10)
    dVdt[0] = F[9] - F[0]
    dVdt[1] = F[0] - F[1]
    dVdt[2] = F[1] - F[2]
    dVdt[3] = F[2] - F[3]
    dVdt[4] = F[3] - F[4]
    dVdt[5] = F[4] - F[5]
    dVdt[6] = F[5] - F[6]
    dVdt[7] = F[6] - F[7]
    dVdt[8] = F[7] - F[8]
    dVdt[9] = F[8] - F[9]

    # Lung cardio to respi model by exchanging the Cardiac output with the lung model
    # Calculate CO as area under the curve of F[0]
    cycle_length = 60 / HR
    cycle_points = int(cycle_length / misc_constants['T'])
    if idx >= cycle_points:
        F0_cycle = F_store[0, idx - cycle_points:idx]
        CO = np.trapz(F0_cycle, dx=misc_constants['T'])   # in ml/sec
    else:
        CO = bloodflows['CO']  # Initial values

    q_p = CO
    sh = 0.02  # Shunt fraction, A 2% anatomical shunt is assumed for blood bypassing the alveoli, appropriate for an adult with no known anatomical abnormalities
    q_Bv = .2 * CO
    q_S = .8 * CO

    # Compute mechanical derivatives
    Ppl_dt = (mechanical_states[0] / (cardio_constants['R_lt'] * C_cw)) - (mechanical_states[1] / (C_cw * cardio_constants['R_lt'])) + Pmus_dt
    dxdt_mechanical = np.dot(A_mechanical, mechanical_states) + np.dot(B_mechanical, [P_ao, Ppl_dt, Pmus_dt])

    # Compute the ventilation rates based on pressures
    Vdot_l = (P_ao - mechanical_states[0]) / cardio_constants['R_ml']
    Vdot_A = (mechanical_states[2] - mechanical_states[3]) / cardio_constants['R_bA']

    p_D_CO2 = FD_CO2 * 713
    p_D_O2 = FD_O2 * 713
    FA_CO2 = p_a_CO2 / 713
    FA_O2 = p_a_O2 / 713

    c_a_CO2 = params['K_CO2'] * p_a_CO2 + params['k_CO2']  # arterial CO2 concentration, in ml_CO2/ml -> 0.5

    # Add a safeguard to ensure the power calculation does not overflow
    if p_a_O2 > 700:
        p_a_O2 = 700

    c_a_O2 = params['K_O2'] * np.power((1 - np.exp(-params['k_O2'] * p_a_O2)), 2)  # Limit p_a_O2 to 700 to prevent overflow

    c_v_CO2 = (c_Scap_CO2)
    c_v_O2 = (c_Scap_O2)

    if misc_constants['MV'] == 1 and P_ao > 6 * 0.735 or misc_constants['MV'] == 0 and mechanical_states[0] < 0:  # Inspiration
        dFD_O2_dt = Vdot_l * 1000 * (FI_O2 - FD_O2) / (gas_exchange_params['V_D'] * 1000)
        dFD_CO2_dt = Vdot_l * 1000 * (FI_CO2 - FD_CO2) / (gas_exchange_params['V_D'] * 1000)

        dp_a_CO2 = (863 * q_p * (1 - sh) * (c_v_CO2 - c_a_CO2) + Vdot_A * 1000 * (p_D_CO2 - p_a_CO2)) / (gas_exchange_params['V_A'] * 1000)
        dp_a_O2 = (863 * q_p * (1 - sh) * (c_v_O2 - c_a_O2) + Vdot_A * 1000 * (p_D_O2 - p_a_O2)) / (gas_exchange_params['V_A'] * 1000)
    else:  # Expiration
        dFD_O2_dt = Vdot_A * 1000 * (FD_O2 - FA_O2) / (gas_exchange_params['V_D'] * 1000)
        dFD_CO2_dt = Vdot_A * 1000 * (FD_CO2 - FA_CO2) / (gas_exchange_params['V_D'] * 1000)

        dp_a_CO2 = 863 * q_p * (1 - bloodflows['sh']) * (c_v_CO2 - c_a_CO2) / (gas_exchange_params['V_A'] * 1000)
        dp_a_O2 = 863 * q_p * (1 - bloodflows['sh']) * (c_v_O2 - c_a_O2) / (gas_exchange_params['V_A'] * 1000)

    # The systemic tissue compartment
    dc_Stis_CO2 = (params['M_S_CO2'] - gas_exchange_params['D_S_CO2'] * (c_Stis_CO2 - c_Scap_CO2)) / params['V_Stis_CO2']  # in ml_CO2 tissue
    dc_Scap_CO2 = (q_S * (c_a_CO2 - c_Scap_CO2) + gas_exchange_params['D_S_CO2'] * (c_Stis_CO2 - c_Scap_CO2)) / params['V_Scap_CO2']  # in ml_CO2 in the capillary
    dc_Stis_O2 = (params['M_S_O2'] - gas_exchange_params['D_S_O2'] * (c_Stis_O2 - c_Scap_O2)) / params['V_Stis_O2']  # in ml_O2 tissue
    dc_Scap_O2 = (q_S * (c_a_O2 - c_Scap_O2) + gas_exchange_params['D_S_O2'] * (c_Stis_O2 - c_Scap_O2)) / params['V_Scap_O2']  # in ml_O2 in the capillary

    # Central control
    u_c = p_a_CO2 - respiratory_control_params['PaCO2_n']
    hr_c = P[0] - cardio_control_params['ABP_n']
    dDelta_Pmus_c = (-Delta_Pmus_c + respiratory_control_params['Gc_A'] * u_c) / respiratory_control_params['tau_c_A']
    dDelta_RR_c = (-Delta_RR_c + respiratory_control_params['Gc_f'] * u_c) / respiratory_control_params['tau_p_f']

    dDelta_HR_c = (-Delta_HR_c + cardio_control_params['Gc_hr'] * hr_c) / cardio_control_params['tau_hr']  # Heart rate
    dDelta_R_c = (-Delta_R_c + cardio_control_params['Gc_r'] * hr_c) / cardio_control_params['tau_r']  # Resistance
    dDelta_UV_c = (-Delta_UV_c + cardio_control_params['Gc_uv'] * hr_c) / cardio_control_params['tau_uv']  # Unstressed volume

    # Combine all derivatives into a single array
    dxdt = np.concatenate([dVdt, dxdt_mechanical, [dFD_O2_dt, dFD_CO2_dt, dp_a_CO2, dp_a_O2,
                                                  dc_Stis_CO2, dc_Scap_CO2, dc_Stis_O2, dc_Scap_O2,
                                                  dDelta_RR_c, dDelta_Pmus_c, driver[1], dDelta_HR_c, dDelta_R_c, dDelta_UV_c]])
    return dxdt

# Ventilator pressure as a function of time (simple square wave for demonstration)
def ventilator_pressure(t):
    RR = respiratory_control_params['RR_0']  # Respiratory Rate (breaths per minute)
    PEEP = float(app.peep_entry.get())  # Positive End-Expiratory Pressure (cm H2O)
    peak_pressure = 20
    TV = .500  # Tidal Volume (ml)
    mode = 'VCV'  # 'VCV' or 'PCV'
    T = 60 / RR  # period of one respiratory cycle in seconds
    IEratio = 1
    TI = T * IEratio / (1 + IEratio)
    TE = T - TI
    exp_time = TE / 5
    cycle_time = t % T
    if mode == 'VCV':
        if 0 <= cycle_time <= TI:
            return (PEEP + 15) * 0.735  # Inhalation , 15 is the pressure support, 1 cmH2O = 0.735 mmHg
        else:
            return PEEP * 0.735  # Exhalation
    elif mode == 'PCV':
        return 20 if cycle_time < 0.5 else PEEP

# Input function for mechanical states
def input_function(t, RR, Pmus_min, IEratio=1.0):
    T = 60 / RR
    TI = T * IEratio / (1 + IEratio)
    TE = T - TI
    exp_time = TE / 5
    cycle_time = t % T

    if 0 <= cycle_time <= TI:
        dPmus_dt = 2 * (-Pmus_min / (TI * TE)) * cycle_time + (Pmus_min * T) / (TI * TE)
    else:
        dPmus_dt = -Pmus_min / (exp_time * (1 - np.exp(-TE / exp_time))) * np.exp(-(cycle_time - TI) / exp_time)

    return np.array([0, dPmus_dt])
import numpy as np




# Initialize state variables for the combined model
initial_state = np.zeros(29)  # initial values are set to zero unless otherwise specified
initial_state[:10] = V  # Initial volumes for cardiovascular model
initial_state[15] = 157 / 731  # Initial FD_O2
initial_state[16] = 7 / 731  # Initial FD_CO2
initial_state[17] = initial_conditions['p_a_CO2']  # Initial p_a_CO2
initial_state[18] = initial_conditions['p_a_O2']  # Initial p_a_O2
initial_state[19] = initial_conditions['c_Stis_CO2']
initial_state[20] = initial_conditions['c_Scap_CO2']
initial_state[21] = initial_conditions['c_Stis_O2']
initial_state[22] = initial_conditions['c_Scap_O2']
initial_state[23] = respiratory_control_params['Delta_RR_c']
initial_state[24] = respiratory_control_params['Delta_Pmus_c']
initial_state[25] = -2  # Initial Pintra

import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.integrate import solve_ivp

class AlarmWindow(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Alarms")
        self.geometry("400x200")
        self.configure(bg='white')  # Set the background color to white
        self.alarm_listbox = tk.Listbox(self, bg='white', fg='black', font=('Helvetica', 16, 'bold'))  # Set text color to black
        self.alarm_listbox.pack(fill=tk.BOTH, expand=True)
        self.alarms = []  # Track alarms with timestamps
        self.last_bp_above_threshold = False  # Track if the last state was above the threshold

    def add_alarm(self, message):
        """Add a new alarm message with a timestamp to the list."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"{timestamp} - {message}"
        if full_message not in self.alarms:
            self.alarms.append(full_message)
            self.alarm_listbox.insert(tk.END, full_message)
            self.alarm_listbox.see(tk.END)  # Automatically scroll to the latest alarm

    def clear_alarms(self):
        """Clear all alarms."""
        self.alarms.clear()
        self.alarm_listbox.delete(0, tk.END)

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ICU Monitor Simulation")
        self.configure(bg='black')
        self.wnd_lngth = 5

        # Initialize the running state
        self.running = False

        # Initialize blood withdrawal state
        self.blood_withdraw_active = False

        # Initialize the alarm state
        self.bp_alarm_triggered = False

        # Create an AlarmWindow instance
        self.alarm_window = AlarmWindow(self)


        # Create a main frame to hold plots and labels
        main_frame = tk.Frame(self, bg='black')
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a grid to align plots and labels
        grid = tk.Frame(main_frame, bg='black')
        grid.pack(fill=tk.BOTH, expand=True)

        # Create a figure for the plots
        self.fig = Figure(figsize=(8, 5), dpi=100, facecolor='black')
        self.canvas = FigureCanvasTkAgg(self.fig, master=grid)

        # ECG Plot
        self.ax_ecg = self.fig.add_subplot(511)
        self.ax_ecg.set_xlim(0, self.wnd_lngth)
        self.ax_ecg.set_ylim(-2, 2)
        self.line_ecg, = self.ax_ecg.plot([], [], color='lime', lw=2)
        self.ax_ecg.set_facecolor('black')
        self.ax_ecg.set_xticks([])
        self.ax_ecg.set_yticks([])
        self.hide_axes(self.ax_ecg)

        self.hr_label = tk.Label(grid, text="HR: 83", font=('Helvetica', 24, 'bold'), fg='lime', bg='black')

        # Blood Pressure Plot
        self.ax_bp = self.fig.add_subplot(512)
        self.ax_bp.set_xlim(0, self.wnd_lngth)
        self.ax_bp.set_ylim(50, 200)
        self.line_bp, = self.ax_bp.plot([], [], color='red', lw=2)
        self.ax_bp.set_facecolor('black')
        self.ax_bp.set_xticks([])
        self.ax_bp.set_yticks([])
        self.hide_axes(self.ax_bp)

        self.bp_label = tk.Label(grid, text="BP: 116/68", font=('Helvetica', 24, 'bold'), fg='red', bg='black')

        # SpO2 Plot
        self.ax_spo2 = self.fig.add_subplot(513)
        self.ax_spo2.set_xlim(0, self.wnd_lngth)
        self.ax_spo2.set_ylim(60, 150)
        self.line_spo2, = self.ax_spo2.plot([], [], color='cyan', lw=2)
        self.ax_spo2.set_facecolor('black')
        self.ax_spo2.set_xticks([])
        self.ax_spo2.set_yticks([])
        self.hide_axes(self.ax_spo2)

        self.spo2_label = tk.Label(grid, text="SpO2: 94", font=('Helvetica', 24, 'bold'), fg='cyan', bg='black')

        # Respiratory Rate Plot
        self.ax_rr = self.fig.add_subplot(514)
        self.ax_rr.set_xlim(0, self.wnd_lngth*5)
        self.ax_rr.set_ylim(-10, 10)
        self.line_rr, = self.ax_rr.plot([], [], 'w')
        self.ax_rr.axis('off')

        self.rr_label = tk.Label(grid, text="RR: 29", font=('Helvetica', 24, 'bold'), fg='white', bg='black')

        # FD_CO2 Plot
        self.ax_gases = self.fig.add_subplot(515)
        self.ax_gases.set_xlim(0, self.wnd_lngth*5)
        self.ax_gases.set_ylim(0, 40)
        self.line_fd_co2, = self.ax_gases.plot([], [], color='yellow', lw=2)
        self.ax_gases.set_facecolor('black')
        self.ax_gases.set_xticks([])
        self.ax_gases.set_yticks([])
        self.hide_axes(self.ax_gases)

        self.etco2_label = tk.Label(grid, text="EtCO2: 35", font=('Helvetica', 24, 'bold'), fg='yellow', bg='black')

        # Arrange the plots and labels in a grid
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=5, sticky="nsew")

        self.hr_label.grid(row=0, column=1, padx=(10,10), sticky="we")
        self.bp_label.grid(row=1, column=1, padx=(10,10), sticky="we")
        self.spo2_label.grid(row=2, column=1, padx=(10,10),sticky="we")
        self.rr_label.grid(row=3, column=1, padx=(10,10),sticky="we")
        self.etco2_label.grid(row=4, column=1, padx=(10,10),sticky="we")

        # Make the grid stretch with the window
        grid.columnconfigure(0, weight=1)
        grid.rowconfigure(0, weight=1)
        grid.rowconfigure(1, weight=1)
        grid.rowconfigure(2, weight=1)
        grid.rowconfigure(3, weight=1)
        grid.rowconfigure(4, weight=1)

        # Open the control window
        self.open_control_window()

        # Initialize time and data arrays
        self.t = 0
        self.dt = 0.02
        self.current_state = initial_state

        self.xdata_ecg = []
        self.ydata_ecg = []

        self.xdata_bp = []
        self.ydata_bp = []

        self.xdata_spo2 = []
        self.ydata_spo2 = []

        self.xdata_rr = []
        self.ydata_rr = []

        self.xdata_gases = []
        self.ydata_fd_co2 = []

        self.volume_to_add = 0

        self.update_plot()

    def hide_axes(self, ax):
        """Hide the axes borders and ticks."""
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')

    def configure_frame(self, frame):
        style = ttk.Style()
        style.configure('My.TFrame', background='black')

    def open_control_window(self):
        """Open a new window with control buttons."""
        control_window = tk.Toplevel(self)
        control_window.title("Control Panel")
        control_window.geometry("300x200")
        control_window.configure(bg='white')

        time_step_label = ttk.Label(control_window, text="Time Step (s)")
        time_step_label.pack(side=tk.TOP, padx=5, pady=5)
        self.time_step_entry = ttk.Entry(control_window)
        self.time_step_entry.pack(side=tk.TOP, padx=5, pady=5)
        self.time_step_entry.insert(0, "0.02")

        self.blood_withdraw_button = ttk.Button(control_window, text="Start Blood Withdrawal", command=self.toggle_blood_withdrawal)
        self.blood_withdraw_button.pack(side=tk.TOP, padx=5, pady=5)

        self.start_button = ttk.Button(control_window, text="Start", command=self.start_animation)
        self.start_button.pack(side=tk.TOP, padx=5, pady=5)

        self.stop_button = ttk.Button(control_window, text="Stop", command=self.stop_animation)
        self.stop_button.pack(side=tk.TOP, padx=5, pady=5)

        volume_label = ttk.Label(control_window, text="Add Volume (ml):")
        volume_label.pack(side=tk.TOP, padx=5, pady=5)
        self.volume_var = tk.StringVar(value="100")
        self.volume_options = ttk.Combobox(control_window, textvariable=self.volume_var, values=["100", "200", "500"])
        self.volume_options.pack(side=tk.TOP, padx=5, pady=5)
        self.volume_button = ttk.Button(control_window, text="Add Volume", command=self.add_volume)
        self.volume_button.pack(side=tk.TOP, padx=5, pady=5)

            # Add options to set PEEP, FiO2, and RR
        peep_label = ttk.Label(control_window, text="PEEP (cmH2O):")
        peep_label.pack(side=tk.TOP, padx=5, pady=5)
        self.peep_var = tk.StringVar(value="5")
        self.peep_entry = ttk.Entry(control_window, textvariable=self.peep_var)
        self.peep_entry.pack(side=tk.TOP, padx=5, pady=5)

        fio2_label = ttk.Label(control_window, text="FiO2 (%):")
        fio2_label.pack(side=tk.TOP, padx=5, pady=5)
        self.fio2_var = tk.StringVar(value="21")
        self.fio2_entry = ttk.Entry(control_window, textvariable=self.fio2_var)
        self.fio2_entry.pack(side=tk.TOP, padx=5, pady=5)

        rr_label = ttk.Label(control_window, text="Respiratory Rate (breaths/min):")
        rr_label.pack(side=tk.TOP, padx=5, pady=5)
        self.rr_var = tk.StringVar(value="12")
        self.rr_entry = ttk.Entry(control_window, textvariable=self.rr_var)
        self.rr_entry.pack(side=tk.TOP, padx=5, pady=5)

        # Button to update ventilator settings
        self.update_settings_button = ttk.Button(control_window, text="Update Settings", command=self.update_ventilator_settings)
        self.update_settings_button.pack(side=tk.TOP, padx=5, pady=5)

        # Add buttons to start/stop the mechanical ventilator
        self.ventilator_on = False  # Initialize ventilator state

        self.start_ventilator_button = ttk.Button(control_window, text="Start Ventilator", command=self.start_ventilator)
        self.start_ventilator_button.pack(side=tk.TOP, padx=5, pady=5)

        self.stop_ventilator_button = ttk.Button(control_window, text="Stop Ventilator", command=self.stop_ventilator)
        self.stop_ventilator_button.pack(side=tk.TOP, padx=5, pady=5)

    def update_ventilator_settings(self):
        """Update the ventilator settings from the control panel."""
        try:
            peep_value = float(self.peep_var.get())
            fio2_value = float(self.fio2_var.get()) / 100.0  # Convert percentage to a decimal
            rr_value = int(self.rr_var.get())

            # Update the PEEP, FiO2, and RR in the corresponding dictionaries
            misc_constants['P_vent'] = peep_value
            gas_exchange_params['FI_O2'] = fio2_value
            respiratory_control_params['RR_0'] = rr_value

            print(f"PEEP set to: {peep_value} cmH2O")
            print(f"FiO2 set to: {fio2_value * 100}%")
            print(f"Respiratory Rate set to: {rr_value} breaths/min")

        except ValueError:
            print("Invalid input. Please enter valid numbers for PEEP, FiO2, and RR.")

    def start_animation(self):
        if not self.running:
            self.running = True
            self.dt = float(self.time_step_entry.get())
            self.update_plot()

    def start_ventilator(self):
        """Start the mechanical ventilator."""
        self.ventilator_on = True
        misc_constants['MV'] = 1  # Update the MV constant to start the ventilator
        print("Ventilator started")  # Optional: for debugging purposes

    def stop_ventilator(self):
        """Stop the mechanical ventilator."""
        self.ventilator_on = False
        misc_constants['MV'] = 0  # Update the MV constant to stop the ventilator
        print("Ventilator stopped")  # Optional: for debugging purposes

    def stop_animation(self):
        self.running = False

    def toggle_blood_withdrawal(self):
        self.blood_withdraw_active = not self.blood_withdraw_active
        if self.blood_withdraw_active:
            self.blood_withdraw_button.config(text="Stop Blood Withdrawal")
        else:
            self.blood_withdraw_button.config(text="Start Blood Withdrawal")

    def add_volume(self):
        try:
            volume = int(self.volume_var.get())
            self.volume_to_add = volume / 10.0  # Divide volume addition over 10 seconds
        except ValueError:
            pass

    def update_plot(self):
        if self.running:
            self.t += self.dt
            t_span = [self.t, self.t + self.dt]
            sol = solve_ivp(extended_state_space_equations, t_span, self.current_state, method='RK45')
            self.current_state = sol.y[:, -1]

            idx = np.searchsorted(misc_constants['t_eval'], self.t)
            if 0 <= idx < len(misc_constants['t_eval']):
                blood_pressure = P_store[0, idx]
                if self.blood_withdraw_active:
                    blood_pressure = 300.0

                self.xdata_bp.append(self.t)
                self.ydata_bp.append(float(blood_pressure))
                self.ax_bp.set_ylim(min(self.ydata_bp) - 10, max(self.ydata_bp) + 10)

                # Check for alarms
                if blood_pressure > 200 and not self.bp_alarm_triggered:
                    self.alarm_window.add_alarm(f"PHILIPSMONITOR - ABPd hoog (1)")
                    self.alarm_window.add_alarm(f"PHILIPSMONITOR - ABPm hoog (1)")
                    self.alarm_window.add_alarm(f"PHILIPSMONITOR - ABPs hoog (1)")
                    self.alarm_window.add_alarm(f"PHILIPSMONITOR - ABP pulseert niet")
                    self.bp_alarm_triggered = True
                elif blood_pressure <= 200:
                    self.bp_alarm_triggered = False


            p_a_O2 = float(sol.y[18, -1])
            CaO2 = (params['K_O2'] * np.power((1 - np.exp(-params['k_O2'] * min(p_a_O2, 700))), 2)) * 100
            Sa_O2 = ((CaO2 - p_a_O2 * 0.003 / 100) / (misc_constants['Hgb'] * 1.34)) * 100

            self.xdata_spo2.append(self.t)
            self.ydata_spo2.append(P_store[1, idx])

            heart_rate = int(HR_store[idx])
            t_segment = np.arange(self.t, self.t + self.dt, self.dt / 10.0)
            ecg_segment = generate_sinus_ecg_segment(t_segment, heart_rate)
            self.xdata_ecg.extend(t_segment)
            self.ydata_ecg.extend(ecg_segment)

            self.line_ecg.set_data(self.xdata_ecg, self.ydata_ecg)
            self.line_bp.set_data(self.xdata_bp, self.ydata_bp)
            self.line_spo2.set_data(self.xdata_spo2, self.ydata_spo2)

            Pl = float(sol.y[14, -1])
            self.xdata_rr.append(self.t)
            self.ydata_rr.append(Pl)
            self.line_rr.set_data(self.xdata_rr, self.ydata_rr)

            FD_CO2 = float(sol.y[16, -1] * 713)
            self.xdata_gases.append(self.t)
            self.ydata_fd_co2.append(FD_CO2)
            self.line_fd_co2.set_data(self.xdata_gases, self.ydata_fd_co2)
            self.ax_gases.collections.clear()
            self.ax_gases.fill_between(self.xdata_gases, self.ydata_fd_co2, where=(np.array(self.ydata_fd_co2) >= 0), color='yellow', alpha=0.3)
            self.ax_gases.fill_between(self.xdata_gases, self.ydata_fd_co2, where=(np.array(self.ydata_fd_co2) <= 0), color='yellow', alpha=0.6)

            self.ax_ecg.set_xlim(self.t - self.wnd_lngth, self.t)
            self.ax_bp.set_xlim(self.t - self.wnd_lngth, self.t)
            self.ax_spo2.set_xlim(self.t - self.wnd_lngth, self.t)
            self.ax_rr.set_xlim(self.t - (self.wnd_lngth*5), self.t)
            self.ax_gases.set_xlim(self.t - (self.wnd_lngth*5), self.t)

            HR = int(HR_store[idx])
            RR = int(respiratory_control_params['RR_0'] + sol.y[23, -1])

            self.hr_label.config(text=f"HR: {HR}")
            self.rr_label.config(text=f"RR: {RR}")

            if len(self.ydata_bp) > 0:
                recent_sap = np.max(self.ydata_bp[-int(5 / self.dt):])
                recent_dap = np.min(self.ydata_bp[-int(5 / self.dt):])
                recent_etco2 = np.max(self.ydata_fd_co2[-int(15 / self.dt):]) # find the max value of the last 15 seconds
            else:
                recent_sap = 0
                recent_dap = 0
                recent_etco2 = 0

            self.bp_label.config(text=f"BP: {int(recent_sap)}/{int(recent_dap)}")
            self.etco2_label.config(text=f"EtCO2: {int(recent_etco2)}")
            self.spo2_label.config(text=f"SpO2: {int(Sa_O2)}")

            if self.volume_to_add > 0:
                add_volume = self.volume_to_add * self.dt
                self.current_state[0] += add_volume / 2
                self.current_state[1] += add_volume / 2
                self.volume_to_add -= add_volume

            self.canvas.draw()

            # Call update_plot function recursively
            self.after(int(self.dt * 1000), self.update_plot)

            # Remove old data to keep the plots clean
            while self.xdata_ecg and self.xdata_ecg[0] < self.t - self.wnd_lngth:
                self.xdata_ecg.pop(0)
                self.ydata_ecg.pop(0)
            while self.xdata_bp and self.xdata_bp[0] < self.t - self.wnd_lngth:
                self.xdata_bp.pop(0)
                self.ydata_bp.pop(0)
            while self.xdata_spo2 and self.xdata_spo2[0] < self.t - self.wnd_lngth:
                self.xdata_spo2.pop(0)
                self.ydata_spo2.pop(0)
            while self.xdata_rr and self.xdata_rr[0] < self.t - (self.wnd_lngth*5):
                self.xdata_rr.pop(0)
                self.ydata_rr.pop(0)
            while self.xdata_gases and self.xdata_gases[0] < self.t - (self.wnd_lngth*5):
                self.xdata_gases.pop(0)
                self.ydata_fd_co2.pop(0)



# Run the application
app = Application() 
app.mainloop()
