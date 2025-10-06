import streamlit as st
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(layout="wide", page_title="Simulador de Cinética Química")

st.title("Simulador de Reacciones Químicas — EDO + Interactividad")

#Sidebar: elegir modelo
st.sidebar.header("Configuración del modelo")
model = st.sidebar.selectbox("Tipo de reacción", [
    "Primer orden: A → B",
    "Segundo orden: 2A → productos",
    "Reversible simple: A ⇌ B",
    "Michaelis-Menten: S → P (cinetica)"
])

#Parámetros globales y tiempo
t_max = st.sidebar.number_input("Tiempo máximo (s)", value=10.0, min_value=0.1, step=0.1)
n_points = st.sidebar.slider("Puntos en la simulación", 50, 2001, 200)
t_eval = np.linspace(0, t_max, n_points)

#Condiciones iniciales y parámetros según modelo
if model == "Primer orden: A → B":
    A0 = st.sidebar.number_input("[A]_0 (M)", value=1.0, min_value=0.0, step=0.1)
    B0 = st.sidebar.number_input("[B]_0 (M)", value=0.0, min_value=0.0, step=0.1)
    k = st.sidebar.number_input("k (s^-1)", value=0.5, min_value=0.0, step=0.01, format="%.6f")
    def rhs(t, y):
        A, B = y
        dA = -k * A
        dB = k * A
        return [dA, dB]
    y0 = [A0, B0]
    species = ["A", "B"]

elif model == "Segundo orden: 2A → productos":
    A0 = st.sidebar.number_input("[A]_0 (M)", value=1.0, min_value=0.0, step=0.1)
    k = st.sidebar.number_input("k (M^-1 s^-1)", value=0.5, min_value=0.0, step=0.01, format="%.6f")
    def rhs(t, y):
        A = y[0]
        dA = -2 * k * A**2
        #opciones: productos no rastreados
        return [dA]
    y0 = [A0]
    species = ["A"]

elif model == "Reversible simple: A ⇌ B":
    A0 = st.sidebar.number_input("[A]_0 (M)", value=1.0, min_value=0.0, step=0.1)
    B0 = st.sidebar.number_input("[B]_0 (M)", value=0.0, min_value=0.0, step=0.1)
    kf = st.sidebar.number_input("k_f (s^-1)", value=1.0, min_value=0.0, step=0.01, format="%.6f")
    kr = st.sidebar.number_input("k_r (s^-1)", value=0.2, min_value=0.0, step=0.01, format="%.6f")
    def rhs(t, y):
        A, B = y
        dA = -kf*A + kr*B
        dB = kf*A - kr*B
        return [dA, dB]
    y0 = [A0, B0]
    species = ["A", "B"]

else:  #Michaelis-Menten
    S0 = st.sidebar.number_input("[S]_0 (M)", value=1.0, min_value=0.0, step=0.1)
    P0 = st.sidebar.number_input("[P]_0 (M)", value=0.0, min_value=0.0, step=0.1)
    Vmax = st.sidebar.number_input("Vmax (M s^-1)", value=1.0, min_value=0.0, step=0.01, format="%.6f")
    Km = st.sidebar.number_input("Km (M)", value=0.5, min_value=0.0, step=0.01, format="%.6f")
    def v_mm(S):
        return Vmax * S / (Km + S) if (Km + S) > 0 else 0.0
    def rhs(t, y):
        S, P = y
        rate = v_mm(S)
        dS = -rate
        dP = rate
        return [dS, dP]
    y0 = [S0, P0]
    species = ["S", "P"]

#Resolver EDO
st.subheader("Simulación numérica")
solver = st.selectbox("Solver SciPy", ["RK45", "Radau", "BDF"], index=0)
sol = solve_ivp(rhs, [0, t_max], y0, t_eval=t_eval, method=solver, atol=1e-8, rtol=1e-6)

#Mostrar resultados
df = pd.DataFrame(sol.y.T, columns=species)
df['t'] = sol.t
df = df[['t'] + species]

col1, col2 = st.columns([2,1])
with col1:
    fig, ax = plt.subplots(figsize=(7,4))
    for s in species:
        ax.plot(df['t'], df[s], label=s)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Concentración (M)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.write("Tabla de resultados (muestras):")
    st.dataframe(df.head(10))

#Subir datos experimentales para ajustar parámetros
st.subheader("Ajuste de parámetros con datos experimentales (opcional)")
st.markdown("Formato CSV esperado: columna `t` con tiempos, y columnas con las especies exactamente como aparecen arriba (ej. `A`, `B`, `S`, `P`).")

uploaded = st.file_uploader("Sube CSV con datos experimentales (opcional)", type=['csv','txt'])
fit_result = None
if uploaded is not None:
    try:
        exp_df = pd.read_csv(uploaded)
        st.write("Datos experimentales (preview):")
        st.dataframe(exp_df.head(10))
        #Interpolamos/extraemos los tiempos experimentales
        t_exp = exp_df['t'].values
        y_exp = np.vstack([exp_df[s].values for s in species]).T  # shape (n, n_species)

        #Definir función de predicción para parámetros a ajustar (depende del modelo)
        if model == "Primer orden: A → B":
            def predict(params):
                k_ = params[0]
                def rhs_p(t, y):
                    A, B = y
                    return [-k_*A, k_*A]
                solp = solve_ivp(rhs_p, [t_exp.min(), t_exp.max()], [exp_df.get('A',df['A'].iloc[0]).iloc[0] if 'A' in exp_df.columns else y0[0], 
                                                                     exp_df.get('B',df['B'].iloc[0]).iloc[0] if 'B' in exp_df.columns else y0[1]], t_eval=t_exp, method=solver)
                return solp.y.T
            x0 = np.array([k])
        elif model == "Segundo orden: 2A → productos":
            def predict(params):
                k_ = params[0]
                def rhs_p(t, y):
                    A = y[0]
                    return [-2*k_*A**2]
                solp = solve_ivp(rhs_p, [t_exp.min(), t_exp.max()], [exp_df.get('A',df['A'].iloc[0]).iloc[0] if 'A' in exp_df.columns else y0[0]], t_eval=t_exp, method=solver)
                return solp.y.T
            x0 = np.array([k])
        elif model == "Reversible simple: A ⇌ B":
            def predict(params):
                kf_, kr_ = params
                def rhs_p(t, y):
                    A, B = y
                    return [-kf_*A + kr_*B, kf_*A - kr_*B]
                icA = exp_df['A'].iloc[0] if 'A' in exp_df.columns else y0[0]
                icB = exp_df['B'].iloc[0] if 'B' in exp_df.columns else y0[1]
                solp = solve_ivp(rhs_p, [t_exp.min(), t_exp.max()], [icA, icB], t_eval=t_exp, method=solver)
                return solp.y.T
            x0 = np.array([kf, kr])
        else:  #Michaelis-Menten
            def predict(params):
                Vmax_, Km_ = params
                def v_mm_local(S):
                    return Vmax_ * S / (Km_ + S) if (Km_ + S) > 0 else 0.0
                def rhs_p(t, y):
                    S, P = y
                    rate = v_mm_local(S)
                    return [-rate, rate]
                icS = exp_df['S'].iloc[0] if 'S' in exp_df.columns else y0[0]
                icP = exp_df['P'].iloc[0] if 'P' in exp_df.columns else y0[1]
                solp = solve_ivp(rhs_p, [t_exp.min(), t_exp.max()], [icS, icP], t_eval=t_exp, method=solver)
                return solp.y.T
            x0 = np.array([Vmax, Km])

        #Residuals para least_squares: aplanar diferencias todas las especies
        def residuals(params):
            y_pred = predict(params)  # shape (n_times, n_species)
            #emparejar shapes y devolver aplanado
            #Si exp_df tiene menos especies, solo usar las que hay
            res = []
            for i, s in enumerate(species):
                if s in exp_df.columns:
                    res.append((y_pred[:, i] - exp_df[s].values))
            return np.hstack(res)

        st.write("Iniciando ajuste numérico...")
        res = least_squares(residuals, x0, bounds=(0, np.inf))
        fit_result = res
        st.success("Ajuste completado.")
        st.write("Parámetros estimados:")
        st.write(res.x)
        st.write(f"RMSE: {np.sqrt(np.mean(res.fun**2)):.4g}")
        # Graficar predicción vs datos
        y_fit = predict(res.x)
        fig2, ax2 = plt.subplots(figsize=(7,4))
        for i, s in enumerate(species):
            if s in exp_df.columns:
                ax2.scatter(t_exp, exp_df[s].values, label=f"{s} (exp)", marker='o')
                ax2.plot(t_exp, y_fit[:, i], label=f"{s} (fit)")
        ax2.set_xlabel("t (s)")
        ax2.set_ylabel("Concentración (M)")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error leyendo/ajustando los datos: {e}")

#Exportar resultados
st.subheader("Exportar resultados")
csv = df.to_csv(index=False)
st.download_button("Descargar CSV de la simulación", csv, file_name="simulacion.csv", mime="text/csv")

st.markdown("**Notas:** usa `Radau` o `BDF` para sistemas rígidos. Ajusta tolerancias si necesitas más precisión.")
