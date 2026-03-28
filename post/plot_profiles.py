from prettyPlot import pretty_plot
from matplotlib.lines import Line2D
from exportAndCrop import export_and_crop
from matplotlib import pyplot as plt
import numpy as np
import os
import time

# Diretório onde o script está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho da pasta raiz do projeto (um nível acima)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))


# Apple palette colors
green = '#61BB46'
red = '#E03A3E'
blue = '#009DDC'
purple = '#963D97'
yellow = '#FDB827'
orange = '#F5821F'
grey = '#818181'
black = '#000000'
color = [blue, orange, green, purple, orange, yellow]


def profile(inputPath, outputPath):


    D = 16
    NX = NZ = 64
    NY = 256
    U_MAX = 0.05



    # Reshape function 
    def vol_reshape(property, inputPath):
        vol_property = (np.fromfile(os.path.join(inputPath, property), dtype=np.float32))
        vol_property_3D = np.reshape(vol_property,(NX,NY,NZ), order = 'F')
        return vol_property_3D
    
    # Find r half function
    def find_r_half(U, r, Uc):
        U_half = Uc / 2
        idx = np.where(U <= U_half)[0][0]
        r1, r2 = r[idx-1], r[idx]
        U1, U2 = U[idx-1], U[idx]
        return r1 + (U_half - U1)*(r2-r1)/(U2-U1)

    # Inputs for budget equation
    uy = "uy_avg.bin"
    # Reshaping properties to 3D
    uy_avg_vol = vol_reshape(uy, inputPath)

    t0 = time.time()
    print(f"Tempo de execução (mesh): {(time.time()-t0)/60:.2f} min")
    print(f"\nDimensões: {NX} x {NY} x {NZ}")

    t0 = time.time()
    uy_c = np.zeros(NY)
    for y in range(NY):
        uy_c[y] = (
            uy_avg_vol[NX//2, y, NZ//2] +
            uy_avg_vol[NX//2-1, y, NZ//2] +
            uy_avg_vol[NX//2, y, NZ//2] +
            uy_avg_vol[NX//2-1, y, NZ//2]
        ) / 4
    uy_c = uy_c[1:-1]
    y = np.arange(NY-2)
    print(f"Tempo de execução (centerline): {(time.time()-t0)/60:.2f} min")


    # AJUSTE DA RETA
    # USER: Ponto inicial e final para o ajuste
    yi_adim = 10
    yf_adim = 15

    #---------------------------------------------------

    t0 = time.time()
    y_adim = y / D
    yi = np.argmin(np.abs(y_adim - yi_adim))
    yf = np.argmin(np.abs(y_adim - yf_adim))

    # Fit for uz centerline

    x_fit = y_adim[yi:yf+1]
    y_fit = (U_MAX/uy_c)[yi:yf+1]


    a, b = np.polyfit(x_fit, y_fit, 1)

    B = 1/a
    y0 = -b * B * D
    print(f"Tempo de execução (ajuste centerline): {(time.time()-t0)/60:.2f} min")

    print(f"\nB = {B}")
    print(f"y0 = {y0}")

    # Margins size according to the overleaf paper format
    PaperWidth  = 612     # points
    MarginPoints = 54     # points
    
    # UZ centerline plot
    fig1, ax1, cbar = pretty_plot(
        xLim=(0, NY/D), yLim=(0, np.ceil(np.max(U_MAX/uy_c))), cLim=(-1, 1),
        plotAspectRatio=(1,1,1),
        xLabel=fr"$y/ D$", yLabel=r"$U_{jet} / u_{y}$",
        yScientificNotation = False, yTickFormat= 2, 
        nxTicks=9, nyTicks=9,
        useColorBar=False,
        paperPoints=PaperWidth,
        marginPoints=MarginPoints,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize = 14,
        dpi=300
    )

    # UZ centerline range
    fig2, ax2, cbar = pretty_plot(
        xLim=(yi_adim, yf_adim), yLim=(((U_MAX/uy_c)[yi]), (U_MAX/uy_c[yf+1])), cLim=(-1, 1),
        plotAspectRatio=(1,1,1),
        xLabel=fr"$y/ D$", yLabel=r"$U_{jet}/u_{y}$",
        yScientificNotation = False, yTickFormat= 2, 
        nxTicks=6, nyTicks=9,
        useColorBar=False,
        paperPoints=PaperWidth,
        marginPoints=MarginPoints,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize = 14,
        dpi=300
    )

    # UZ centerline fit 
    fig3, ax3, cbar = pretty_plot(
        xLim=(yi_adim, yf_adim), yLim=(((U_MAX/uy_c)[yi]), (U_MAX/uy_c[yf+1])), cLim=(-1, 1),
        plotAspectRatio=(1,1,1),
        xLabel=fr"$y/ D$", yLabel=r"$U_{jet}/u_{y}$",
        yScientificNotation = False, yTickFormat= 2, 
        nxTicks=6, nyTicks=9,
        useColorBar=False,
        paperPoints=PaperWidth,
        marginPoints=MarginPoints,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize = 14,
        dpi=300
    )

    ax1.yaxis.set_label_coords(-0.125, 0.5)
    ax2.yaxis.set_label_coords(-0.125, 0.5)
    ax3.yaxis.set_label_coords(-0.125, 0.5)

    # UZ centerline plot
    ax1.plot(y/D, U_MAX/uy_c,
            ls = '-', lw = 3, color = black, markersize=5)

    
    # UZ centerline range
    ax2.plot(y_adim[yi:yf+1],(U_MAX/uy_c)[yi:yf+1], marker='o', mfc = 'none', ms = 2,
            ls = 'none', lw = 3, color = black)


    # UZ centerline fit  
    ax3.plot(y_adim[yi:yf+1], (U_MAX/uy_c)[yi:yf+1], marker='o', mfc = 'none', ms = 2,
            ls = 'none', lw = 3, color = black,
            label='LBM')
    
    ax3.plot(x_fit, a*x_fit + b,
            ls = '--', lw = 3, color = red, markersize=5,
            label='$(z-z_0)/(BD)$')

    ax3.legend(fontsize = 14)

    export_and_crop(fig1, f"{outputPath}/uz.png")
    export_and_crop(fig2, f"{outputPath}/uz_c_range.png")
    export_and_crop(fig3, f"{outputPath}/uz_c_fit.png")

    # Radial profile
    fig4, ax4, cbar = pretty_plot(
        xLim=(0, 3), yLim=(0, 0.4), cLim=(-1, 1),
        plotAspectRatio=(1,1,1),
        xLabel=r"$r / D$", yLabel=r"$<u_z>/U_{jet}$",
        yScientificNotation = False, 
        xTickFormat = 1, yTickFormat= 1, 
        nxTicks=7, nyTicks=11,
        useColorBar=False,
        paperPoints=PaperWidth,
        marginPoints=MarginPoints,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize = 14,
        dpi=300
    )

    # Radial self-similar r/D
    fig5, ax5, cbar = pretty_plot(
        xLim=(0, 0.3), yLim=(0, 1), cLim=(-1, 1),
        plotAspectRatio=(1,1,1),
        xLabel=r"$r/(z-z_0)$", yLabel=r"$<u_z>/U_0$",
        yScientificNotation = False, 
        xTickFormat = 2, yTickFormat= 1, 
        nxTicks=7, nyTicks = 6,
        useColorBar=False,
        paperPoints=PaperWidth,
        marginPoints=MarginPoints,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize = 14,
        dpi=300
    )

    # Radial self-similar r_half/D
    fig6, ax6, cbar = pretty_plot(
        xLim=(0, 3), yLim=(0, 1), cLim=(-1, 1),
        plotAspectRatio=(1,1,1),
        xLabel=r'$r/r_{1/2}$', yLabel=r'$<u_z>/U_0$',
        yScientificNotation = False, 
        xTickFormat = 2, yTickFormat= 1, 
        nxTicks = 7, nyTicks = 6,
        useColorBar=False,
        paperPoints=PaperWidth,
        marginPoints=MarginPoints,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=90,
        useGrid=False,
        fontSize = 14,
        dpi=300
    )
    ax4.yaxis.set_label_coords(-0.125, 0.585)
    ax5.yaxis.set_label_coords(-0.125, 0.585)
    ax6.yaxis.set_label_coords(-0.125, 0.585)


    # PERFIL RADIAL (linha x)

    # USER: Slices de interesse
    slices = np.array([10,11,12,13,14,15])  # em D

    xc = NX//2
    zc = NZ//2

    x = np.arange(NX)

    cycle = ['o', 's', '^', 'D', 'v', '<', '>']

    r_half = np.zeros(slices.size)
    S = np.zeros(slices.size)

    t0 = time.time()

    for j in range(slices.size):

        y = int(slices[j]*D)

        # linha ao longo de x passando pelo centro
        uy_line = uy_avg_vol[:, y, zc]

        # coordenada radial
        r_line = x - xc

        # usar apenas metade positiva
        mask = r_line >= 0
        r = r_line[mask]
        uy_r = uy_line[mask]


        # calcular r_half
        r_half[j] = find_r_half(uy_r, r, uy_c[y])
        S[j] = r_half[j] / (y - y0)
        print(f"Slice {slices[j]}D → r_half = {r_half[j]:.4f} | S = {S[j]:.4f}")

        # ---------------- VELOCIDADE RADIAL ----------------

        ax4.plot(r/D, uy_r/U_MAX, lw = 3, ls = 'none',
        marker=cycle[j], ms = 3, mfc = 'none', color = color[j], mec = black, label = f'z = {slices[j]}D')
        ax4.legend(fontsize = 10)

        ax5.plot(r/(y-y0), uy_r/uy_c[y], lw = 3, ls = 'none',
        marker=cycle[j], ms = 3, color = color[j], mfc = 'none', mec = black, label = f'z = {slices[j]}D')
        print("Qtde pontos:", len(uy_r/uy_c[y]))

        ax6.plot(r/r_half[j], uy_r/uy_c[y], lw = 3, ls = 'none',
        marker=cycle[j], ms = 3, color = color[j], mfc = 'none', mec = black, label = f'z = {slices[j]}D')
        ax6.legend(fontsize = 10)

    print(f"\nTempo de execução (self-similar): {(time.time()-t0)/60:.2f} min")

    export_and_crop(fig4, f"{outputPath}/uz_r.png")




    # FIG 2 – Hussein
    xH = np.array([
    0,0.010599057494345044,0.019815647572123837,0.030875562697158625,
    0.04009215277493742,0.04838709790833872,0.056221185411050346,
    0.06405527291376191,0.07281104051935197,0.08479260058876421,
    0.09400919066654304,0.10737325331102249,0.11658984338880131,
    0.12949308356109202,0.14147464363050433,0.15345620369991658,
    0.16866359139165193,0.18248847650832015,0.19723500656936588,
    0.21290321673329005,0.22672810184995826,0.2447004595333271
    ])

    yH = np.array([
    0.9986970883174464,0.988273596045439,0.9622149647712097,
    0.9205211944947582,0.8657980588782979,0.8136807963298389,
    0.7563517876453765,0.7042345250969178,0.6469055164124551,
    0.5635178764537632,0.514006536676201,0.4332247200826162,
    0.38371338030505414,0.32899019498569865,0.2690553629361291,
    0.2247556698887811,0.16742676061010808,0.11791512261517773,
    0.08143319788051967,0.05276854442960405,0.03452768146806465,
    0.024104189196056974
    ])

    ax5.plot(xH, yH,
        ls = 'none', marker='x', ms = 4, color = red, label = 'Hussein et al. (1994)')
    ax5.legend(fontsize = 10)
    
    
    export_and_crop(fig5, f"{outputPath}/uz_selfsimilar.png")
    export_and_crop(fig6, f"{outputPath}/uz_rhalf.png")







if __name__ == "__main__":
    import sys

    if len(sys.argv) < 1:
        print("Uso: python3 plot_profiles.py ")
        sys.exit(1)

    inputPath = os.path.join(PROJECT_ROOT, "JET_VTK")
    outputPath = os.path.join("resultsJET")
    os.makedirs(outputPath, exist_ok=True)

    profile(
        inputPath = inputPath,
        outputPath = outputPath,
    )