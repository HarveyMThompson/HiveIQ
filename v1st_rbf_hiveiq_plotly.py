# program to calculate rbf response surface given DoE points and 
# calculating the single and multi-objective optimisation from these responses
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

rbfmodel = 1     # Gaussian weights
n = 50
ndv = 2
x1min_actual = 0.0
x1max_actual = 1.0
x2min_actual = 40.0
x2max_actual = 180.0
max_pd = 33.56574
min_pd = 26.7848
nbetas = 101

##########################################################
# reading data into relevant files
##########################################################
def read_data():
    
    st.session_state.Xr = np.loadtxt("data/Xr.txt")
    st.session_state.Yr = np.loadtxt("data/Yr.txt")
    st.session_state.Zr_rbf = np.loadtxt("data/Zr_rbf.txt")
    st.session_state.x_scaled = np.loadtxt("data/x_scaled.txt")
    st.session_state.x = np.loadtxt("data/x.txt")
    st.session_state.pressure_drop = np.loadtxt("data/pressure_drop.txt")
    st.session_state.beta_min = np.loadtxt("data/beta_min.txt")
    st.session_state.beta_array = np.loadtxt("data/beta_array.txt")
    st.session_state.RMSE_array = np.loadtxt("data/RMSE_array.txt")
    lam1 =  np.loadtxt("data/lam.txt")
    st.session_state.lam = np.transpose(np.asmatrix(lam1))
    st.session_state.xopt = np.loadtxt("data/xopt.txt")
    st.session_state.yopt = np.loadtxt("data/yopt.txt")
    st.session_state.optval = np.loadtxt("data/optval.txt")

    # st.session_state.x = st.session_state.x_actual[:,0]
    # st.session_state.y = st.session_state.x_actual[:,1]
    st.session_state.z = st.session_state.pressure_drop
    st.session_state.title_text = 'RBF Surrogate Model of Heat Flux (W)'
    
    # process the data into a form that can be used with the go.Mesh3d function
    xmesh =[]
    ymesh = []
    zmesh = []
    for i in range(41):
        for j in range(40):
            xmesh.append(st.session_state.Xr[i,j])
            ymesh.append(st.session_state.Yr[i,j])
            zmesh.append(st.session_state.Zr_rbf[i,j])

    st.session_state.xmesh = xmesh
    st.session_state.ymesh = ymesh
    st.session_state.zmesh = zmesh
    

# plot out the RBF surrogate model of pressure drop
def pd_plotly_plot():
 
    x = st.session_state.xmesh
    y = st.session_state.ymesh
    z = st.session_state.zmesh
    trace2 = go.Mesh3d(x=x,
                       y=y,
                       z=z,
                       opacity=0.5,
                       color='rgba(244,22,100,0.6)')

    xscatter = st.session_state.x[:,0]
    yscatter = st.session_state.x[:,1]
    zscatter = st.session_state.pressure_drop
    trace3 = go.Scatter3d(x=xscatter, y=yscatter, z=zscatter, mode='markers', marker=dict(size=3))
    data2 = [trace2,trace3]
    layout = go.Layout(title=st.session_state.title_text,
                       title_font=dict(size=20,
                                       color='blue',
                                       family='Arial'),
                       title_x=0.25,
                       title_y=0.85)

    fig2 = go.Figure(data=data2, layout=layout)
    fig2.update_scenes(xaxis=dict(title="Area coefficient",nticks=6, range=[0.0,1.0]),
                       yaxis=dict(title="Wall thickness (mm)",nticks=10, range=[40.0,180.0]), 
                       zaxis=dict(title="Heat flux (W)"))

    st.plotly_chart(fig2)    

# end pd_plot    
 
# plot out the RBF surrogate model of pressure drop
def pd_matplotlib_plot():
    
    # Plot out RBF approximation
    plt.figure()
    Xr = st.session_state.Xr
    Yr = st.session_state.Yr
    Zr_rbf = st.session_state.Zr_rbf
    x = st.session_state.x
    pressure_drop = st.session_state.pressure_drop
    xopt = st.session_state.xopt
    yopt = st.session_state.yopt
    optval = st.session_state.optval
    
    plt.suptitle('RBF Surrogate Model of Heat Flux (W)')
    #ax = fig.gca(projection='3d')
    ax=plt.axes(projection='3d')
    surf = ax.plot_surface(Xr, Yr, Zr_rbf, rstride=8, cstride=8, alpha=0.3, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)

    ax.set_xlim(0, 1)
    ax.set_ylim(40, 180)

    # plot out the scatter points
    ax.scatter(x[:,0],x[:,1],pressure_drop, c='r', marker='o',s=4)
    # ax.scatter(xopt,yopt,optval, c='k', marker='o',s=16)   # plot optimum point
    ax.set_xlabel('Area coefficient')
    ax.set_ylabel('Wall thickness (mm)')
    ax.set_zlabel('Heat flux (W)')
    st.pyplot(plt)
    
# calculate pressure drop at specified point
def pd_f(x): 
    return rbf_point(n,st.session_state.x_scaled,ndv,x,st.session_state.lam,rbfmodel,st.session_state.beta_min)

# calculate surrogate models of thermal resistance and pressure drop
def calc_surrogates(x1,x2):

    xp = np.zeros(ndv)
    x1_scaled = (x1-x1min_actual)/(x1max_actual-x1min_actual)
    x2_scaled = (x2-x2min_actual)/(x2max_actual-x2min_actual)
    xp[0] = x1_scaled
    xp[1] = x2_scaled
    pd_val = pd_f(xp)   
    return pd_val


st.title("CFD Analysis of Heat Flux from Hives")
st.write("This application enables you to explore the heat flux from a hive")

checking_password = 0
if (checking_password != 0):
    if 'pwdcheck' not in st.session_state:
        st.session_state['pwdcheck'] = 0
        password_guess = st.text_input('What is the password?')
        if password_guess != st.secrets["password"]:
            st.stop()

# read in data for calculations only at the beginning of session
if 'Xr' not in st.session_state:
    read_data()

tab1, tab2, tab3 = st.tabs(["Surrogate model", "Heat flux plot (matplotlib)", "Heat flux plot (plotly)"])
with tab1:
    
    # Create the input sliders
    row1 = st.columns([1,1])

    default_value = 0.2
    x1 = row1[0].slider("Area coefficient",x1min_actual,x1max_actual,0.5)
    x2 = row1[1].slider("Wall thickness (mm)",x2min_actual,x2max_actual,100.0)
    pd_val = calc_surrogates(x1,x2)
    st.write(f'Heat flux: {pd_val:.1f} W')

with tab2:
    pd_matplotlib_plot()

with tab3:
    pd_plotly_plot()


