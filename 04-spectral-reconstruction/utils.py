import colour
import numpy as np
import matplotlib.pyplot as plt
import io
from scipy.interpolate import interp1d
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from uvicorn import Server, Config
from base64 import b64encode
from PIL import Image
from IPython.display import display_html, IFrame


async def hsi_observer(hsi: np.ndarray,
                       wavelengths: np.ndarray,
                       host='127.0.0.1',
                       port=8800,
                       slice_width=40,
                       slice_height=40,
                       slice_step=4,
                       iframe_width=800,
                       iframe_height=300):
    """
    Usage in Jupyter Notebook:
    ```
    # hsi shape: (H, W, n)
    await hsi_observer(hsi, wavelengths, host=<notebook ip address>)
    ```
    """

    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    sens = interp1d(cmfs.wavelengths, cmfs.values, axis=0, bounds_error=False, fill_value=0)(wavelengths)
    xyz = hsi @ sens
    rgb = colour.XYZ_to_sRGB(xyz)

    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    jpeg_buf = io.BytesIO()
    Image.fromarray(rgb).save(jpeg_buf, 'JPEG')
    base64_data = str(b64encode(jpeg_buf.getvalue()), encoding='ascii')

    app = FastAPI()

    @app.get('/', response_class=HTMLResponse)
    async def overview():
        html = template.format(
            base64_data=base64_data,
            slice_width=slice_width,
            slice_height=slice_height,
            slice_step=slice_step
        )
        return HTMLResponse(html)

    @app.get('/plot/{ly}:{ry}:{sy},{lx}:{rx}:{sx}', response_class=Response)
    async def plot(ly: int, ry: int, sy: int, lx: int, rx: int, sx: int):
        fig, ax = plt.subplots()
        n_bands = hsi.shape[-1]
        spectra = hsi[ly:ry:sy, lx:rx:sx].reshape(-1, n_bands).T
        ax.plot(wavelengths, spectra, c='k', alpha=0.3)
        ax.set_xlabel('Wavelength, nm')
        ax.grid()
        jpeg_buf = io.BytesIO()
        fig.savefig(jpeg_buf, format='jpeg')
        plt.close(fig)
        return Response(content=jpeg_buf.getvalue(), media_type="image/jpeg")

    display_html(IFrame(f'http://{host}:{port}', iframe_width, iframe_height))
    config = Config(app, host='0.0.0.0', port=port, log_level='critical')
    await Server(config).serve()


template = '''
<head>
    <style>
        html, body {{
            padding: 0;
            margin: 0;
            width: 100%;
            height: 100%;
            background: white;
            overflow: hidden;
        }}
        body {{
            display: flex;
            flex-direction: column;
        }}
        .plots {{
            flex: 1;
            display: flex;
            overflow: hidden;
        }}
        #base {{
            width: 50%;
            object-fit: contain;
            object-position: top left;
        }}
        #spectra {{
            width: 50%;
            object-fit: contain;
            object-position: top left;
            border: none;
            outline: none;
        }}
        .hidden {{
            position: fixed;
            top: 100%;
            left: 100%;
        }}
        .rect {{
            position: absolute;
            border: 1px solid white;
            box-shadow: 0 0 3px;
            opacity: 0.5;
            box-sizing: content-box;
            display: none;
        }}
    </style>
    <script>
        let actualWidth, actualHeight;
        function measureImage(image) {{
            console.log('test')
            actualWidth = image.width
            actualHeight = image.height
        }}
    </script>
</head>
<body>
    <div class="plots">
        <img src="data:image/jpeg;base64,{base64_data}" id="base">
        <img id="spectra">
    </div>
    <div class="controls"><!-- future something --></div>
    <div class="hidden">
        <img src="data:image/jpeg;base64,{base64_data}" id="measure" onload="measureImage(this)">
    </div>
    <div class="rect"></div>

    <script>
        const step = Number("{slice_step}")
        const rect = document.querySelector(".rect")
        const baseImage = document.querySelector("#base")
        const out = document.querySelector("#spectra")
        window.onmousemove = function(event) {{
            const {{client: [lx, rx, ly, ry]}} = calcSlice(event)
            if (lx == rx || ly == ry) {{
                rect.style.display = "none"
            }} else {{
                rect.style.display = "block"
            }}
            console.log(lx, rx, ly, ry)
            rect.style.left = lx
            rect.style.top = ly
            rect.style.width = rx - lx
            rect.style.height = ry - ly
        }}
        window.onclick = function(event) {{
            if (rect.style.display == "none") {{
                return
            }}
            const {{image: [lx, rx, ly, ry]}} = calcSlice(event)
            out.src = `/plot/${{ly}}:${{ry}}:4,${{lx}}:${{rx}}:4`
        }}
        window.onmouseover = function() {{
            rect.style.display = "block"
        }}
        window.onmouseout = function() {{
            rect.style.display = "none"
        }}
        const sliceWidth = Number("{slice_width}"), sliceHeight = Number("{slice_height}")
        function calcSlice(event) {{
            console.log(sliceWidth, sliceHeight)
            const scale = Math.min(baseImage.width / actualWidth, baseImage.height / actualHeight)
            console.log(baseImage.width, baseImage.height, actualWidth, actualHeight)
            const x = Math.floor((event.pageX - baseImage.offsetLeft) / scale)
            const y = Math.floor((event.pageY - baseImage.offsetTop) / scale)
            const lx = clip(x - Math.floor(sliceWidth / 2), 0, actualWidth)
            const rx = clip(x - Math.floor(sliceWidth / 2) + sliceWidth, 0, actualWidth)
            const ly = clip(y - Math.floor(sliceHeight / 2), 0, actualHeight)
            const ry = clip(y - Math.floor(sliceHeight / 2) + sliceHeight, 0, actualHeight)
            return {{
                image: [lx, rx, ly, ry],
                client: [
                    lx * scale + baseImage.offsetLeft,
                    rx * scale + baseImage.offsetLeft,
                    ly * scale + baseImage.offsetTop,
                    ry * scale + baseImage.offsetTop
                ]
            }}
        }}
        function clip(x, l, r) {{
            return Math.max(l, Math.min(r, x))
        }}
    </script>
</body>
'''