# CROPA E NÃO CENTRALIZA


import matplotlib.pyplot as plt


def export_and_crop(fig, file_name, vertical_crop=True, dpi=600, transparent=False, pad_inches=0.01):
    """
    Exporta uma figura Matplotlib e faz crop vertical opcional, similar à função MATLAB exportAndCrop.
    
    Args:
        fig : matplotlib.figure.Figure
            Figura a ser exportada.
        file_name : str
            Nome do arquivo de saída (ex: 'figura.png').
        vertical_crop : bool, default True
            Se True, remove espaços verticais em branco após salvar a figura.
        dpi : int, default 600
            Resolução da figura em dpi.
        transparent : bool, default False
            Se True, salva com fundo transparente.
        pad_inches : float, default 0.01
            Padding extra para bbox_inches='tight'.
    """
    
    # Exportar figura sem crop automático (equivalente a "-nocrop" do MATLAB)
    fig.savefig(file_name, dpi=dpi, transparent=transparent)
    
    # Se vertical_crop for True, reaplica bbox_inches tight para remover espaços verticais
    if vertical_crop:
        fig.savefig(file_name, dpi=dpi, transparent=transparent, bbox_inches='tight', pad_inches=pad_inches)
    
    # Fecha a figura (equivalente a close all no MATLAB)
    plt.close(fig)
    
