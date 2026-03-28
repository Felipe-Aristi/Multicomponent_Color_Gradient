import numpy as np
import vtk
from vtk.util import numpy_support

NX = 64
NY = 256
NZ = 64

arquivo_bin = "../JET_VTK/uy_avg.bin"
arquivo_vti = "../JET_VTK/uy_avg.vti"

data = np.fromfile(arquivo_bin, dtype=np.float32)

esperado = NX * NY * NZ
if data.size != esperado:
    raise ValueError(
        f"Tamanho incompatível: esperado {esperado} valores, mas o arquivo tem {data.size}"
    )

print(f"Lido: {data.size} valores")
print(f"min = {data.min():.6e}, max = {data.max():.6e}")

data = np.ascontiguousarray(data)

vtk_array = numpy_support.numpy_to_vtk(
    num_array=data,
    deep=True,
    array_type=vtk.VTK_FLOAT
)
vtk_array.SetName("uy_avg")

image = vtk.vtkImageData()
image.SetDimensions(NX, NY, NZ)
image.SetSpacing(1.0, 1.0, 1.0)
image.SetOrigin(0.0, 0.0, 0.0)
image.GetPointData().SetScalars(vtk_array)

writer = vtk.vtkXMLImageDataWriter()
writer.SetFileName(arquivo_vti)
writer.SetInputData(image)

# mais seguro para depuração
writer.SetDataModeToAscii()

ok = writer.Write()
if ok != 1:
    raise RuntimeError("Falha ao escrever o arquivo VTI")

print(f"Arquivo salvo em: {arquivo_vti}")