from paddleocr import PaddleOCR
from osgeo import gdal, ogr, osr
import numpy as np
import json
import os

# 启用 GDAL 异常处理
gdal.UseExceptions()

# 设置 GDAL 字符编码为 UTF-8
gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')
gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'YES')

# 打印PaddlePaddle版本，检查GPU情况
import paddle
print(paddle.__version__)  
paddle.utils.run_check()
# 通过 text_detection_model_dir 指定本地模型路径
# pipeline = PaddleOCR(text_detection_model_dir="./your_det_model_path")
# 默认使用 PP-OCRv5_server_det 模型作为默认文本检测模型，如果微调的不是该模型，通过 text_detection_model_name 修改模型名称
# pipeline = PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det", text_detection_model_dir="./your_v5_mobile_det_model_path")

# 使用 GDAL 读取 TIFF 文件
dataset = gdal.Open("D:\\data\\beijing_diming\\测试数据\\L13\\114_36_111_34_TIF_L13.tif")
if dataset is None:
    raise Exception("无法打开 TIFF 文件")

# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    # 指定检测模型路径
    det_model_dir="C:\\Users\\Administrator\\.paddlex\\official_models\\PP-OCRv5_server_det",  # 文本检测模型目录
    # 指定识别模型路径
    rec_model_dir="C:\\Users\\Administrator\\.paddlex\\official_models\\PP-OCRv5_server_rec",  # 文本识别模型目录
    # 指定分类模型路径（如果需要）
    # cls_model_dir="./models/cls_model",
    # 或者通过模型名称指定
    # text_detection_model_name="PP-OCRv5_server_det",
    # text_detection_model_dir="./models/your_det_model_path"
)

# 获取波段数和图像数据
bands = dataset.RasterCount
width = dataset.RasterXSize
height = dataset.RasterYSize

# 获取地理转换参数（用于像素坐标到地理坐标的转换）
geotransform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

print(f"图像尺寸: {width} x {height}, 波段数: {bands}")
print(f"地理转换参数: {geotransform}")
print(f"投影信息: {projection}")

# 滑窗参数
window_size = 2048  # 减小滑窗大小，让每个窗口包含更少文字
overlap = 100  # 重叠区域,避免文字被切断
stride = window_size - overlap  # 实际步长
min_window_size = 100  # 最小窗口尺寸,跳过过小的边缘窗口

# 存储所有识别结果
all_results = []

# 像素坐标转换为地理坐标的函数
def pixel_to_geo(x, y, geotransform, img_height):
    """
    将像素坐标转换为地理坐标
    注意: GDAL的Y轴方向通常是向下的(geotransform[5]为负值)
    """
    geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return geo_x, geo_y

def export_to_shapefile(results, projection, output_path):
    """
    将 OCR 识别结果导出为 Shapefile
    
    参数:
        results: 识别结果列表，每个元素包含 text, confidence, geo_box
        projection: 投影信息（WKT 格式）
        output_path: 输出 Shapefile 路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建空间参考系统
    srs = osr.SpatialReference()
    if projection and projection.strip():
        # 如果有投影信息，使用原始影像的投影
        try:
            srs.ImportFromWkt(projection)
        except Exception as e:
            print(f"警告: 投影信息导入失败 ({e})，使用 WGS84 地理坐标系")
            srs.ImportFromEPSG(4326)  # 默认使用 WGS84
    else:
        print("警告: 无投影信息，使用 WGS84 地理坐标系")
        srs.ImportFromEPSG(4326)  # 默认使用 WGS84
    
    # 创建 Shapefile 驱动
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    # 如果文件已存在，先删除
    if os.path.exists(output_path):
        driver.DeleteDataSource(output_path)
    
    # 创建数据源
    datasource = driver.CreateDataSource(output_path)
    layer = datasource.CreateLayer('ocr_results', srs, ogr.wkbPolygon)
    
    # 添加属性字段
    field_text = ogr.FieldDefn('name', ogr.OFTString)
    field_text.SetWidth(254)
    layer.CreateField(field_text)
    
    field_conf = ogr.FieldDefn('scores', ogr.OFTReal)
    field_conf.SetPrecision(4)
    layer.CreateField(field_conf)
    
    # 写入每个识别结果
    for result in results:
        # 创建要素
        feature = ogr.Feature(layer.GetLayerDefn())
        
        # 创建多边形几何对象
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in result['geo_box']:
            ring.AddPoint(point[0], point[1])
        # 闭合多边形
        ring.AddPoint(result['geo_box'][0][0], result['geo_box'][0][1])
        
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        
        # 设置几何和属性
        feature.SetGeometry(polygon)
        feature.SetField('name', result['text'])
        feature.SetField('scores', result['confidence'])
        
        # 创建要素
        layer.CreateFeature(feature)
        
        # 清理
        feature = None
    
    # 关闭数据源
    datasource = None
    
    # 创建 .cpg 文件指定编码
    cpg_path = output_path.replace('.shp', '.cpg')
    with open(cpg_path, 'w') as f:
        f.write('UTF-8')
    
    print(f"Shapefile 已保存到: {output_path}")

# 遍历图像的滑窗
for y in range(0, height, stride):
    for x in range(0, width, stride):
        # 计算当前窗口的实际尺寸
        current_width = min(window_size, width - x)
        current_height = min(window_size, height - y)
        
        print(f"处理窗口: x={x}, y={y}, width={current_width}, height={current_height}")
        
        # 读取当前窗口的图像数据
        if bands == 1:
            # 单波段灰度图像
            band = dataset.GetRasterBand(1)
            window_array = band.ReadAsArray(x, y, current_width, current_height)
        elif bands >= 3:
            # 多波段彩色图像，读取前三个波段 (RGB)
            window_array = np.zeros((current_height,current_width, 3), dtype=np.uint8)
            for i in range(3):
                band = dataset.GetRasterBand(i + 1)
                band_data = band.ReadAsArray(x, y, current_width, current_height)
                window_array[:, :, i] = band_data
        else:
            raise Exception("不支持的波段数")
        
        # 确保数组数据类型为 uint8
        if window_array.dtype != np.uint8:
            # 归一化到 0-255 范围
            img_min = window_array.min()
            img_max = window_array.max()
            if img_max > img_min:
                window_array = ((window_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                window_array = window_array.astype(np.uint8)
        
        # 对当前窗口执行 OCR 推理
        result = ocr.predict(input=window_array)
        
        # 处理识别结果，调整坐标到全图坐标系
        for res in result:
            # 检查结果是否包含识别数据
            if 'rec_texts' not in res or not res['rec_texts']:
                continue
            
            res.save_to_img("data/output/window_{}_{}.png".format(x, y))

            # 遍历所有识别到的文本
            for text, score, box in zip(res['rec_texts'], res['rec_scores'], res['rec_boxes']):
                # box 格式为 [min_x, min_y, max_x, max_y]
                # 调整坐标到全图坐标系（加上窗口偏移）
                min_x = box[0] + x
                min_y = box[1] + y
                max_x = box[2] + x
                max_y = box[3] + y
                
                # 将像素坐标转换为地理坐标
                geo_min = pixel_to_geo(min_x, min_y, geotransform, height)
                geo_max = pixel_to_geo(max_x, max_y, geotransform, height)
                
                # 构建矩形框的四个角点（地理坐标）
                geo_box = [
                    [geo_min[0], geo_min[1]],  # 左下
                    [geo_max[0], geo_min[1]],  # 右下
                    [geo_max[0], geo_max[1]],  # 右上
                    [geo_min[0], geo_max[1]]   # 左上
                ]
                
                # 存储结果
                all_results.append({
                    "text": text,
                    "confidence": float(score),
                    "geo_box": geo_box
                })
                
                print(f"  识别到文字: {text} (置信度: {score:.4f})")


# 导出 Shapefile
output_shp_path = "data/output/ocr_results.shp"
export_to_shapefile(all_results, projection, output_shp_path)
# 关闭数据集
dataset = None

print(f"\n识别完成！共识别到 {len(all_results)} 个文本")
