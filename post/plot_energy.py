from prettyPlot import pretty_plot
from matplotlib.lines import Line2D
from exportAndCrop import export_and_crop 
from matplotlib.ticker import FixedLocator
import numpy as np
import os

# Diretório onde o script está
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print(BASE_DIR)

# Caminho da pasta raiz do projeto (um nível acima)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

print(PROJECT_ROOT)

# Apple palette colors
green = '#61BB46'
red = '#E03A3E'
blue = '#009DDC'
purple = '#963D97'
yellow = '#FDB827'
orange = '#F5821F'
color =[green, red, blue, purple, yellow, orange]



def totalKineticEnergy(property, field, inputPath, outputPath):

    D = 16
    U_MAX = 0.05
    interval = 2000
    # energy = f"{property}.bin"
    energy = f"tke_{field}.bin"
    tke = np.fromfile(os.path.join(inputPath, energy), dtype=np.float32)
    if field != "diff":
        tke = tke / 9*(U_MAX**2)
        minYLim = np.min(tke)*0.95
        maxYLim = np.max(tke)*1.05
    else:
        minYLim = 1e-6
        maxYLim = 1e-2
    t_starmax = (len(tke)*interval)*U_MAX / D
    t_star = np.linspace(0,t_starmax, len(tke))
    


    if property == "kinetic_energy" and field != "diff":
        property_label = r"$ E_{K}^{*}$"
        maxLim = t_starmax
        minLim = 0
    else:
        property_label = r"$ E_{K}^{*}$"
        minLim = t_starmax/2
        maxLim = t_starmax
    

    # Margins size according to the overleaf article format
    PaperWidth  = 612     # points
    MarginPoints = 54     # points

    fig, ax, cbar = pretty_plot(
        xLim = (minLim, maxLim), yLim = (minYLim, maxYLim), cLim=(-1, 1),
        plotAspectRatio=(1,1,1),
        xLabel=r"$t^{*}$", yLabel = property_label,
        yScientificNotation = True, yTickFormat= 2,
        nxTicks = 5, nyTicks=9, 
        useColorBar=False,
        paperPoints=PaperWidth,
        marginPoints=MarginPoints,
        textWidth=1,
        boxMarginScale=0.085,
        yLabelAngle=0,
        useGrid=True,
        fontSize = 14,
        dpi=300
    )


    ax.yaxis.set_label_coords(-0.125, 0.5)


    # ------------------------------- Plot for UX_CY ----------------------------------------------
    # print(len(t_star))
    ax.plot(t_star, tke,
            ls = '-' ,lw = 3, color = blue, markersize=5, label = "256")
    if field == "diff":
        ax.set_yscale('log')


    ax.grid(True)
    ax.legend(loc = 'lower right', title = "Grid", title_fontsize = 18, fontsize= 18) 
    # plt.show()
    export_and_crop(fig, f"{outputPath}/{property}_.png")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Uso: python3 plot_energy.py <property (kinetic_energy)> <field (total/avg/diff)>")
        sys.exit(1)

    property = sys.argv[1]
    field = sys.argv[2]

    inputPath = os.path.join(PROJECT_ROOT, "JET_VTK")
    outputPath = os.path.join("resultsJET")
    os.makedirs(outputPath, exist_ok=True)

    totalKineticEnergy(
        property = property,
        field = field,
        inputPath = inputPath,
        outputPath = outputPath,
    )