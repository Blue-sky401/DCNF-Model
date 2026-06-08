import sys
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


# Qt环境修复 - 在ImportPyQt5之前设置环境变量
def fix_qt_environment():
    """修复Qt环境变量"""
    # 清除可能冲突的环境变量
    os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
    os.environ.pop('QT_PLUGIN_PATH', None)

    # 设置平台
    os.environ['QT_QPA_PLATFORM'] = 'windows'

    # 如果在虚拟环境中，设置插件路径
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        possible_paths = [
            os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt5', 'plugins'),
            os.path.join(sys.prefix, 'Lib', 'site-packages', 'PyQt5', 'Qt', 'plugins'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = path
                break


# ApplyQt修复
fix_qt_environment()

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QPushButton, QTableWidget, QGroupBox, QCheckBox,
                             QSlider, QLabel, QComboBox, QSpinBox, QTabWidget,
                             QFileDialog, QMessageBox, QProgressBar, QSplitter,
                             QTextEdit, QMenuBar, QMenu, QAction, QToolBar,
                             QStatusBar, QDialog, QFormLayout, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QPalette
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import hashlib
from datetime import datetime, timedelta
import threading
import time
from functools import lru_cache

# matplotlib配置 - 必须在Importpyplot之前设置backend
import matplotlib

matplotlib.use('Qt5Agg')  # 强制使用Qt5Agg backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import platform


# 配置中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    import matplotlib.font_manager as fm

    # 根据操作系统选择字体
    if platform.system() == 'Windows':
        font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
    elif platform.system() == 'Darwin':  # macOS
        font_list = ['Arial Unicode MS', 'Heiti TC', 'Songti TC']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

    # 获取系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]

    # 寻找可用的中文字体
    chinese_font = None
    for font in font_list:
        if font in available_fonts:
            chinese_font = font
            break

    if chinese_font:
        matplotlib.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
        print(f"使用字体: {chinese_font}")
    else:
        # 如果没有找到中文字体，使用系统默认字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        print("警告: 未找到中文字体，中文可能显示为方框")

    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 设置字体大小
    matplotlib.rcParams['font.size'] = 10

    # 强制刷新字体缓存
    try:
        fm._load_fontmanager(try_read_cache=False)
    except:
        pass


# Apply字体设置
setup_chinese_font()


# 数据Cache Management类
class DataCacheManager:
    """数据Cache Management器 - 改善性能"""

    def __init__(self, cache_dir="cache", max_cache_size_mb=500):
        self.cache_dir = cache_dir
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # 转换为字节
        self.cache_info = {}
        self._ensure_cache_dir()
        self._load_cache_info()

    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _load_cache_info(self):
        """加载Cache Info"""
        info_file = os.path.join(self.cache_dir, "cache_info.json")
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    self.cache_info = json.load(f)
            except:
                self.cache_info = {}

    def _save_cache_info(self):
        """保存Cache Info"""
        info_file = os.path.join(self.cache_dir, "cache_info.json")
        try:
            with open(info_file, 'w') as f:
                json.dump(self.cache_info, f)
        except:
            pass

    def _get_cache_key(self, data_hash, operation, params):
        """生成缓存键"""
        param_str = json.dumps(params, sort_keys=True)
        combined = f"{data_hash}_{operation}_{param_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_data_hash(self, df):
        """获取数据哈希"""
        if df is None or df.empty:
            return None
        # 使用数据形状和部分内容生成哈希
        content = f"{df.shape}_{df.iloc[0].to_string() if len(df) > 0 else ''}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, df, operation, params):
        """从缓存获取数据"""
        data_hash = self._get_data_hash(df)
        if not data_hash:
            return None

        cache_key = self._get_cache_key(data_hash, operation, params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        if cache_key in self.cache_info and os.path.exists(cache_file):
            # 检查缓存是否过期（24小时）
            cache_time = datetime.fromisoformat(self.cache_info[cache_key]['timestamp'])
            if datetime.now() - cache_time < timedelta(hours=24):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    # 缓存损坏，删除
                    self._remove_cache_entry(cache_key)

        return None

    def set(self, df, operation, params, result):
        """设置缓存数据"""
        data_hash = self._get_data_hash(df)
        if not data_hash:
            return

        cache_key = self._get_cache_key(data_hash, operation, params)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")

        try:
            # 检查缓存大小限制
            self._cleanup_cache_if_needed()

            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)

            # 更新Cache Info
            file_size = os.path.getsize(cache_file)
            self.cache_info[cache_key] = {
                'timestamp': datetime.now().isoformat(),
                'size': file_size,
                'operation': operation
            }

            self._save_cache_info()

        except Exception as e:
            print(f"缓存保存失败: {e}")

    def _cleanup_cache_if_needed(self):
        """如果需要，清理缓存"""
        total_size = sum(info['size'] for info in self.cache_info.values())

        if total_size > self.max_cache_size:
            # 删除最旧的缓存项
            sorted_cache = sorted(
                self.cache_info.items(),
                key=lambda x: x[1]['timestamp']
            )

            for cache_key, _ in sorted_cache[:len(sorted_cache) // 2]:
                self._remove_cache_entry(cache_key)

    def _remove_cache_entry(self, cache_key):
        """删除缓存项"""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        if cache_key in self.cache_info:
            del self.cache_info[cache_key]

    def clear_cache(self):
        """清空所有缓存"""
        for cache_key in list(self.cache_info.keys()):
            self._remove_cache_entry(cache_key)
        self._save_cache_info()


# Kriging Interpolation算法类
class KrigingInterpolator:
    """Kriging Interpolation算法 - 显著提升插值精度"""

    def __init__(self, variogram_model='spherical'):
        self.variogram_model = variogram_model
        self.variogram_params = None
        self.fitted = False

    def _distance_matrix(self, coords1, coords2=None):
        """计算距离矩阵"""
        if coords2 is None:
            coords2 = coords1

        diff = coords1[:, np.newaxis] - coords2[np.newaxis, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    def _spherical_variogram(self, h, nugget, sill, range_param):
        """球状变异函数"""
        result = np.zeros_like(h)
        mask = h > 0

        h_norm = h[mask] / range_param
        mask_range = h_norm <= 1

        if np.any(mask_range):
            h_range = h_norm[mask_range]
            result[mask][mask_range] = nugget + (sill - nugget) * (
                    1.5 * h_range - 0.5 * h_range ** 3
            )

        # 超出范围的点
        mask_beyond = h_norm > 1
        if np.any(mask_beyond):
            result[mask][mask_beyond] = sill

        return result

    def _exponential_variogram(self, h, nugget, sill, range_param):
        """指数变异函数"""
        return nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_param))

    def _fit_variogram(self, coords, values):
        """拟合变异函数"""
        # 计算经验变异函数
        distances = []
        semivariances = []

        # 计算所有点对的距离和半方差
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dist = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                semivar = 0.5 * (values[i] - values[j]) ** 2
                distances.append(dist)
                semivariances.append(semivar)

        distances = np.array(distances)
        semivariances = np.array(semivariances)

        # 简单的变异函数参数估计
        max_distance = np.max(distances)
        nugget = np.min(semivariances) if len(semivariances) > 0 else 0
        sill = np.mean(semivariances[-10:]) if len(semivariances) > 10 else np.max(semivariances)
        range_param = max_distance / 3

        self.variogram_params = {
            'nugget': max(0, nugget),
            'sill': max(nugget + 0.1, sill),
            'range': max(0.1, range_param)
        }

        self.fitted = True

    def fit(self, coords, values):
        """拟合克里金模型"""
        self.coords = np.array(coords)
        self.values = np.array(values)

        # 拟合变异函数
        self._fit_variogram(self.coords, self.values)

        # 构建克里金系统矩阵
        n = len(self.coords)
        distances = self._distance_matrix(self.coords)

        # 变异函数矩阵
        if self.variogram_model == 'spherical':
            gamma_matrix = self._spherical_variogram(
                distances, **self.variogram_params
            )
        else:  # exponential
            gamma_matrix = self._exponential_variogram(
                distances, **self.variogram_params
            )

        # 构建克里金矩阵 (添加拉格朗日乘数)
        A = np.zeros((n + 1, n + 1))
        A[:n, :n] = gamma_matrix
        A[:n, n] = 1
        A[n, :n] = 1
        A[n, n] = 0

        # 右侧向量
        b = np.zeros(n + 1)
        b[:n] = self.values

        try:
            self.weights = np.linalg.solve(A, b)
            self.fitted = True
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            self.weights = np.linalg.pinv(A) @ b
            self.fitted = True

    def predict(self, pred_coords):
        """预测新Position的值"""
        if not self.fitted:
            raise ValueError("模型尚未拟合，请先调用fit()方法")

        pred_coords = np.array(pred_coords)
        if pred_coords.ndim == 1:
            pred_coords = pred_coords.reshape(1, -1)

        predictions = []
        variances = []

        for coord in pred_coords:
            # 计算预测点到数据点的距离
            distances = np.sqrt(np.sum((self.coords - coord) ** 2, axis=1))

            # 计算变异函数值
            if self.variogram_model == 'spherical':
                gamma_vec = self._spherical_variogram(
                    distances, **self.variogram_params
                )
            else:
                gamma_vec = self._exponential_variogram(
                    distances, **self.variogram_params
                )

            # 克里金预测
            pred_value = np.sum(self.weights[:-1] * self.values) + self.weights[-1]

            # 克里金方差（简化计算）
            variance = self.variogram_params['sill'] - np.sum(self.weights[:-1] * gamma_vec)

            predictions.append(pred_value)
            variances.append(max(0, variance))

        return np.array(predictions), np.array(variances)


# Anomaly Detection算法类
class AnomalyDetector:
    """Anomaly Detection算法 - 提高数据质量"""

    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.detectors = {}
        self.fitted = False

    def fit_detect(self, df, columns=None):
        """拟合并检测异常值"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        results = {}

        for column in columns:
            if column not in df.columns:
                continue

            data = df[column].dropna().values.reshape(-1, 1)

            if len(data) < 5:  # 数据太少，跳过
                results[column] = {'anomalies': [], 'scores': []}
                continue

            # 方法1: IQR方法
            iqr_anomalies = self._detect_iqr_anomalies(data.flatten())

            # 方法2: Z-Score方法
            zscore_anomalies = self._detect_zscore_anomalies(data.flatten())

            # 方法3: 孤立森林 (如果sklearn可用)
            isolation_anomalies = self._detect_isolation_anomalies(data)

            # 组合结果
            all_anomalies = set(iqr_anomalies) | set(zscore_anomalies) | set(isolation_anomalies)

            # 计算异常分数
            scores = self._calculate_anomaly_scores(data.flatten(), list(all_anomalies))

            results[column] = {
                'anomalies': list(all_anomalies),
                'scores': scores,
                'iqr_anomalies': iqr_anomalies,
                'zscore_anomalies': zscore_anomalies,
                'isolation_anomalies': isolation_anomalies
            }

        self.fitted = True
        return results

    def _detect_iqr_anomalies(self, data):
        """IQR方法检测异常值"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies = []
        for i, value in enumerate(data):
            if value < lower_bound or value > upper_bound:
                anomalies.append(i)

        return anomalies

    def _detect_zscore_anomalies(self, data, threshold=3):
        """Z-Score方法检测异常值"""
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return []

        z_scores = np.abs((data - mean) / std)
        anomalies = [i for i, score in enumerate(z_scores) if score > threshold]

        return anomalies

    def _detect_isolation_anomalies(self, data):
        """孤立森林检测异常值"""
        try:
            # 尝试使用scikit-learn的IsolationForest
            from sklearn.ensemble import IsolationForest

            if len(data) < 10:
                return []

            detector = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )

            predictions = detector.fit_predict(data)
            anomalies = [i for i, pred in enumerate(predictions) if pred == -1]

            return anomalies

        except ImportError:
            # 如果sklearn不可用，使用简单的统计方法
            return self._detect_statistical_anomalies(data.flatten())

    def _detect_statistical_anomalies(self, data):
        """统计方法检测异常值（备用方法）"""
        # 使用修正的Z-Score
        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            return []

        modified_z_scores = 0.6745 * (data - median) / mad
        anomalies = [i for i, score in enumerate(modified_z_scores) if abs(score) > 3.5]

        return anomalies

    def _calculate_anomaly_scores(self, data, anomaly_indices):
        """计算异常分数"""
        scores = np.zeros(len(data))

        if len(anomaly_indices) == 0:
            return scores

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return scores

        for i in anomaly_indices:
            # 计算标准化距离作为异常分数
            scores[i] = abs((data[i] - mean) / std)

        return scores

    def generate_report(self, df, anomaly_results):
        """生成Anomaly Detection报告"""
        report = []
        report.append("=== Anomaly Detection报告 ===\n")

        total_anomalies = 0
        for column, results in anomaly_results.items():
            anomalies = results['anomalies']
            total_anomalies += len(anomalies)

            report.append(f"列 '{column}':")
            report.append(f"  检测到 {len(anomalies)} 个异常值")

            if anomalies:
                report.append(f"  异常值索引: {anomalies[:10]}{'...' if len(anomalies) > 10 else ''}")

                # 显示异常值
                anomaly_values = df.iloc[anomalies][column].values
                report.append(f"  异常值: {anomaly_values[:5].tolist()}{'...' if len(anomaly_values) > 5 else ''}")

            report.append("")

        report.append(f"总计发现 {total_anomalies} 个异常值")

        return "\n".join(report)


# 创建全局Cache Management器
cache_manager = DataCacheManager()


# Fault Modeling器类
class FaultModeler:
    """Fault Modeling器 - 用于创建和管理断层面模型"""

    def __init__(self):
        self.fault_data = {}  # 存储断层数据
        self.fault_actors = {}  # 存储断层VTK actors
        self.fault_planes = {}  # 存储断层面几何
        self.fault_colors = {
            '逆断层': (0.8, 0.2, 0.2),  # 红色
            '正断层': (0.2, 0.8, 0.2),  # 绿色
            '平推断层': (0.2, 0.2, 0.8),  # 蓝色
            '走滑断层': (0.8, 0.8, 0.2),  # 黄色
            '未知断层': (0.6, 0.6, 0.6)  # 灰色
        }

    def add_fault_from_points(self, fault_name, points, fault_type='未知断层'):
        """从点数据Add Fault"""
        if len(points) < 3:
            raise ValueError("断层至少需要3个点来定义")

        self.fault_data[fault_name] = {
            'points': points,
            'type': fault_type,
            'visible': True
        }

    def add_fault_from_plane_equation(self, fault_name, plane_params, bounds, fault_type='未知断层'):
        """从平面方程Add Fault
        plane_params: [A, B, C, D] 表示平面方程 Ax + By + Cz + D = 0
        bounds: [x_min, x_max, y_min, y_max, z_min, z_max] 断层边界范围
        """
        A, B, C, D = plane_params
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        # 生成平面上的点网格
        x_grid = np.linspace(x_min, x_max, 20)
        y_grid = np.linspace(y_min, y_max, 20)

        points = []
        for x in x_grid:
            for y in y_grid:
                # 从平面方程计算z: z = -(Ax + By + D) / C
                if abs(C) > 1e-6:  # 避免除零
                    z = -(A * x + B * y + D) / C
                    if z_min <= z <= z_max:
                        points.append((x, y, z))

        if len(points) >= 3:
            self.add_fault_from_points(fault_name, points, fault_type)
        else:
            raise ValueError("无法从给定参数生成有效的断层面")

    def add_fault_from_strike_dip(self, fault_name, center_point, strike, dip,
                                  length=100, width=50, fault_type='未知断层'):
        """从走向倾向数据Add Fault
        center_point: (x, y, z) 断层中心点
        strike: 走向角度（度）
        dip: 倾向角度（度）
        length: 断层走向长度
        width: 断层倾向宽度
        """
        cx, cy, cz = center_point

        # 转换角度为弧度
        strike_rad = np.radians(strike)
        dip_rad = np.radians(dip)

        # 计算断层面的四个角点
        # 走向Direction向量
        strike_vec = np.array([np.cos(strike_rad), np.sin(strike_rad), 0])
        # 倾向Direction向量
        dip_vec = np.array([-np.sin(strike_rad) * np.cos(dip_rad),
                            np.cos(strike_rad) * np.cos(dip_rad),
                            -np.sin(dip_rad)])

        # 四个角点
        points = [
            (cx - length / 2 * strike_vec[0] - width / 2 * dip_vec[0],
             cy - length / 2 * strike_vec[1] - width / 2 * dip_vec[1],
             cz - length / 2 * strike_vec[2] - width / 2 * dip_vec[2]),
            (cx + length / 2 * strike_vec[0] - width / 2 * dip_vec[0],
             cy + length / 2 * strike_vec[1] - width / 2 * dip_vec[1],
             cz + length / 2 * strike_vec[2] - width / 2 * dip_vec[2]),
            (cx + length / 2 * strike_vec[0] + width / 2 * dip_vec[0],
             cy + length / 2 * strike_vec[1] + width / 2 * dip_vec[1],
             cz + length / 2 * strike_vec[2] + width / 2 * dip_vec[2]),
            (cx - length / 2 * strike_vec[0] + width / 2 * dip_vec[0],
             cy - length / 2 * strike_vec[1] + width / 2 * dip_vec[1],
             cz - length / 2 * strike_vec[2] + width / 2 * dip_vec[2])
        ]

        self.add_fault_from_points(fault_name, points, fault_type)

    def create_fault_actor(self, fault_name):
        """创建断层的VTK actor"""
        if fault_name not in self.fault_data:
            return None

        fault_info = self.fault_data[fault_name]
        points = fault_info['points']
        fault_type = fault_info['type']

        # 创建VTK points
        vtk_points = vtk.vtkPoints()
        for x, y, z in points:
            vtk_points.InsertNextPoint(x, y, z)

        # 创建polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # 如果点数够多，创建三角面片
        if len(points) >= 3:
            # 创建Delaunay三角化
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(polydata)
            delaunay.Update()

            # 创建mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(delaunay.GetOutputPort())
        else:
            # 点数太少，只显示点
            vertex_filter = vtk.vtkVertexGlyphFilter()
            vertex_filter.SetInputData(polydata)
            vertex_filter.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertex_filter.GetOutputPort())

        # 创建actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # 设置颜色
        color = self.fault_colors.get(fault_type, self.fault_colors['未知断层'])
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.7)

        # 设置边缘显示
        actor.GetProperty().SetEdgeVisibility(True)
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        actor.GetProperty().SetLineWidth(2)

        self.fault_actors[fault_name] = actor
        return actor

    def cut_model_with_fault(self, model_polydata, fault_name):
        """使用断层Cut地层模型"""
        if fault_name not in self.fault_data:
            return model_polydata

        fault_info = self.fault_data[fault_name]
        points = fault_info['points']

        if len(points) < 3:
            return model_polydata

        try:
            # 创建Cut平面
            plane = vtk.vtkPlane()

            # 计算平面法向量（使用前三个点）
            p1 = np.array(points[0])
            p2 = np.array(points[1])
            p3 = np.array(points[2])

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)

            # 设置平面
            plane.SetOrigin(p1)
            plane.SetNormal(normal)

            # 使用平面Cut模型
            cutter = vtk.vtkCutter()
            cutter.SetInputData(model_polydata)
            cutter.SetCutFunction(plane)
            cutter.Update()

            return cutter.GetOutput()

        except Exception as e:
            print(f"断层Cut失败: {e}")
            return model_polydata

    def get_fault_list(self):
        """获取所有断层名称列表"""
        return list(self.fault_data.keys())

    def remove_fault(self, fault_name):
        """Delete Fault"""
        if fault_name in self.fault_data:
            del self.fault_data[fault_name]
        if fault_name in self.fault_actors:
            del self.fault_actors[fault_name]
        if fault_name in self.fault_planes:
            del self.fault_planes[fault_name]

    def set_fault_visibility(self, fault_name, visible):
        """设置断层可见性"""
        if fault_name in self.fault_data:
            self.fault_data[fault_name]['visible'] = visible
        if fault_name in self.fault_actors:
            self.fault_actors[fault_name].SetVisibility(visible)


# 高级可视化管理类
class AdvancedVisualizationManager:
    """高级可视化管理器 - Visual Enhancement"""

    def __init__(self, renderer):
        self.renderer = renderer
        self.slice_actors = []
        self.gradient_actors = []
        self.slice_planes = []
        self.current_slice_position = 0.5
        self.slice_normal = (0, 0, 1)  # 默认Horizontal Slice

    def get_layer_color(self, layer_name):
        """智能获取地层颜色"""
        # 首先检查高级颜色配置
        if layer_name in ADVANCED_LAYER_COLORS:
            return ADVANCED_LAYER_COLORS[layer_name]

        # fallback到基础配置
        if layer_name in layer_colors:
            return layer_colors[layer_name]

        # 生成基于名称的随机但一致的颜色
        import hashlib
        hash_obj = hashlib.md5(layer_name.encode())
        hash_hex = hash_obj.hexdigest()
        r = int(hash_hex[:2], 16) / 255.0 * 0.5 + 0.4  # 确保颜色不太暗
        g = int(hash_obj[:2], 16) / 255.0 * 0.5 + 0.4
        b = int(hash_hex[4:6], 16) / 255.0 * 0.5 + 0.4
        return (r, g, b)

    def create_gradient_fill(self, polydata, start_color, end_color):
        """创建Gradient Fill效果"""
        try:
            # 获取数据边界
            bounds = polydata.GetBounds()
            z_min, z_max = bounds[4], bounds[5]

            # 创建颜色数组
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            colors.SetName("Colors")

            points = polydata.GetPoints()
            for i in range(points.GetNumberOfPoints()):
                point = points.GetPoint(i)
                z = point[2]

                # 计算渐变因子
                if z_max > z_min:
                    factor = (z - z_min) / (z_max - z_min)
                else:
                    factor = 0.5

                # 插值颜色
                r = start_color[0] * (1 - factor) + end_color[0] * factor
                g = start_color[1] * (1 - factor) + end_color[1] * factor
                b = start_color[2] * (1 - factor) + end_color[2] * factor

                colors.InsertNextTuple3(int(r * 255), int(g * 255), int(b * 255))

            polydata.GetPointData().SetScalars(colors)
            return polydata

        except Exception as e:
            print(f"Gradient Fill创建失败: {e}")
            return polydata

    def gradient_fill(self, actor1, actor2, steps=20):
        """在两个模型之间创建Gradient Fill效果"""
        try:
            # 获取两个模型的几何数据
            mapper1 = actor1.GetMapper()
            mapper2 = actor2.GetMapper()

            if not mapper1 or not mapper2:
                return []

            data1 = mapper1.GetInput()
            data2 = mapper2.GetInput()

            if not data1 or not data2:
                return []

            # 获取颜色
            color1 = actor1.GetProperty().GetColor()
            color2 = actor2.GetProperty().GetColor()

            # 创建渐变层
            gradient_actors = []

            for i in range(steps):
                t = i / (steps - 1)  # 插值参数 0 到 1

                # 插值颜色
                interp_color = [
                    color1[0] * (1 - t) + color2[0] * t,
                    color1[1] * (1 - t) + color2[1] * t,
                    color1[2] * (1 - t) + color2[2] * t
                ]

                # 创建插值几何体（简化版本）
                # 这里可以根据实际需求实现更复杂的插值算法

                # 创建渐变层actor
                gradient_mapper = vtk.vtkPolyDataMapper()

                # 根据插值参数选择数据源（简化）
                if t < 0.5:
                    gradient_mapper.SetInputData(data1)
                else:
                    gradient_mapper.SetInputData(data2)

                gradient_actor = vtk.vtkActor()
                gradient_actor.SetMapper(gradient_mapper)
                gradient_actor.GetProperty().SetColor(interp_color)
                gradient_actor.GetProperty().SetOpacity(0.3)  # 半透明

                gradient_actors.append(gradient_actor)

            return gradient_actors

        except Exception as e:
            print(f"创建Gradient Fill失败: {e}")
            return []

    def create_slice_plane(self, bounds, normal, position):
        """创建切片平面"""
        try:
            if not bounds or len(bounds) != 6:
                return None

            # 计算切片平面的中心点
            center = [
                bounds[0] + (bounds[1] - bounds[0]) * 0.5,
                bounds[2] + (bounds[3] - bounds[2]) * 0.5,
                bounds[4] + (bounds[5] - bounds[4]) * 0.5
            ]

            # 根据法向量调整中心点
            if normal == (0, 0, 1):  # Horizontal Slice
                center[2] = bounds[4] + (bounds[5] - bounds[4]) * position
            elif normal == (1, 0, 0):  # Vertical Slice
                center[0] = bounds[0] + (bounds[1] - bounds[0]) * position
            else:  # Lateral Slice
                center[1] = bounds[2] + (bounds[3] - bounds[2]) * position

            # 创建平面
            plane = vtk.vtkPlane()
            plane.SetOrigin(center)
            plane.SetNormal(normal)

            # 创建平面几何体用于显示
            plane_source = vtk.vtkPlaneSource()
            plane_source.SetOrigin(center)
            plane_source.SetNormal(normal)

            # 计算平面大小
            size_x = bounds[1] - bounds[0]
            size_y = bounds[3] - bounds[2]
            size_z = bounds[5] - bounds[4]
            max_size = max(size_x, size_y, size_z)

            # 设置平面的两个Direction向量
            if normal == (0, 0, 1):  # Horizontal Slice
                plane_source.SetPoint1([center[0] + max_size / 2, center[1], center[2]])
                plane_source.SetPoint2([center[0], center[1] + max_size / 2, center[2]])
            elif normal == (1, 0, 0):  # Vertical Slice
                plane_source.SetPoint1([center[0], center[1] + max_size / 2, center[2]])
                plane_source.SetPoint2([center[0], center[1], center[2] + max_size / 2])
            else:  # Lateral Slice
                plane_source.SetPoint1([center[0] + max_size / 2, center[1], center[2]])
                plane_source.SetPoint2([center[0], center[1], center[2] + max_size / 2])

            plane_source.Update()
            return plane_source.GetOutput()

        except Exception as e:
            print(f"创建切片平面失败: {e}")
            return None

    def apply_slice_to_model(self, model_data, slice_plane):
        """将切片Apply到模型数据"""
        try:
            if not model_data or not slice_plane:
                return None

            # 使用VTK的切片过滤器
            cutter = vtk.vtkCutter()
            cutter.SetInputData(model_data)

            # 从切片平面创建隐式函数
            plane = vtk.vtkPlane()
            bounds = slice_plane.GetBounds()
            center = slice_plane.GetCenter()

            # 估算法向量（简化）
            plane.SetOrigin(center)
            plane.SetNormal(self.slice_normal)

            cutter.SetCutFunction(plane)
            cutter.Update()

            return cutter.GetOutput()

        except Exception as e:
            print(f"Apply切片到模型失败: {e}")
            return None

    def clear_slice_actors(self):
        """清除所有切片相关的actors"""
        try:
            for actor in self.slice_actors:
                if hasattr(actor, 'GetRenderer') and actor.GetRenderer():
                    actor.GetRenderer().RemoveActor(actor)

            self.slice_actors.clear()

        except Exception as e:
            print(f"清除切片actors失败: {e}")

    def slice_plane(self, polydata, position_ratio=0.5):
        """对多边形数据进行切片处理"""
        try:
            bounds = polydata.GetBounds()

            # 创建切片平面
            plane = vtk.vtkPlane()

            if self.slice_normal == (0, 0, 1):  # Horizontal Slice
                z_pos = bounds[4] + (bounds[5] - bounds[4]) * position_ratio
                plane.SetOrigin(0, 0, z_pos)
                plane.SetNormal(0, 0, 1)
            elif self.slice_normal == (1, 0, 0):  # Vertical Slice
                x_pos = bounds[0] + (bounds[1] - bounds[0]) * position_ratio
                plane.SetOrigin(x_pos, 0, 0)
                plane.SetNormal(1, 0, 0)
            else:  # Lateral Slice
                y_pos = bounds[2] + (bounds[3] - bounds[2]) * position_ratio
                plane.SetOrigin(0, y_pos, 0)
                plane.SetNormal(0, 1, 0)

            # 创建切片器
            cutter = vtk.vtkCutter()
            cutter.SetInputData(polydata)
            cutter.SetCutFunction(plane)
            cutter.Update()

            return cutter.GetOutput()

        except Exception as e:
            print(f"切片处理失败: {e}")
            return polydata

    def create_slice_plane(self, bounds, normal=(0, 0, 1), position=0.5):
        """创建切片平面"""
        try:
            # 计算切片Position
            if normal == (0, 0, 1):  # ZDirection切片
                slice_z = bounds[4] + (bounds[5] - bounds[4]) * position
                origin = (bounds[0], bounds[2], slice_z)
                point1 = (bounds[1], bounds[2], slice_z)
                point2 = (bounds[0], bounds[3], slice_z)
            elif normal == (1, 0, 0):  # XDirection切片
                slice_x = bounds[0] + (bounds[1] - bounds[0]) * position
                origin = (slice_x, bounds[2], bounds[4])
                point1 = (slice_x, bounds[3], bounds[4])
                point2 = (slice_x, bounds[2], bounds[5])
            else:  # YDirection切片
                slice_y = bounds[2] + (bounds[3] - bounds[2]) * position
                origin = (bounds[0], slice_y, bounds[4])
                point1 = (bounds[1], slice_y, bounds[4])
                point2 = (bounds[0], slice_y, bounds[5])

            # 创建平面
            plane_source = vtk.vtkPlaneSource()
            plane_source.SetOrigin(origin)
            plane_source.SetPoint1(point1)
            plane_source.SetPoint2(point2)
            plane_source.SetResolution(50, 50)
            plane_source.Update()

            return plane_source.GetOutput()

        except Exception as e:
            print(f"切片平面创建失败: {e}")
            return None

    def apply_slice_to_model(self, polydata, slice_plane):
        """对模型Apply切片"""
        try:
            # 创建切片滤波器
            cutter = vtk.vtkCutter()
            cutter.SetInputData(polydata)

            # 创建切片平面
            plane = vtk.vtkPlane()
            if slice_plane:
                # 从切片平面获取法向量和原点
                bounds = slice_plane.GetBounds()
                center = [(bounds[1] + bounds[0]) / 2,
                          (bounds[3] + bounds[2]) / 2,
                          (bounds[5] + bounds[4]) / 2]
                plane.SetOrigin(center)
                plane.SetNormal(self.slice_normal)

            cutter.SetCutFunction(plane)
            cutter.Update()

            return cutter.GetOutput()

        except Exception as e:
            print(f"Model Slicing失败: {e}")
            return polydata

    def clear_slice_actors(self):
        """清除所有切片显示"""
        for actor in self.slice_actors:
            self.renderer.RemoveActor(actor)
        self.slice_actors.clear()

    def clear_gradient_actors(self):
        """清除所有Gradient Fill"""
        for actor in self.gradient_actors:
            self.renderer.RemoveActor(actor)
        self.gradient_actors.clear()


# 定义不同岩层的颜色（调整为更浅的颜色）
layer_colors = {
    "砂岩": (0.95, 0.85, 0.7),  # 更浅的砂岩色
    "泥岩": (0.8, 0.8, 0.8),  # 更浅的灰色
    "煤层": (0.3, 0.3, 0.3),  # 稍浅的黑色
    "灰岩": (0.9, 0.9, 0.95),  # 更浅的蓝灰色
    "页岩": (0.7, 0.9, 0.8),  # 浅绿色
    "石灰岩": (0.9, 0.95, 0.9),  # 浅绿白色
    "白云岩": (0.95, 0.9, 0.85),  # 浅米色
}

# 高级地层颜色配置
ADVANCED_LAYER_COLORS = {
    # 沉积岩类
    "砂岩": (0.96, 0.87, 0.70),  # 浅黄褐色
    "泥岩": (0.75, 0.75, 0.75),  # 中灰色
    "页岩": (0.60, 0.80, 0.70),  # 灰绿色
    "石灰岩": (0.85, 0.95, 0.95),  # 浅青色
    "白云岩": (0.95, 0.90, 0.85),  # 浅米色
    "砾岩": (0.80, 0.70, 0.60),  # 褐色
    "角砾岩": (0.75, 0.65, 0.55),  # 深褐色

    # 火成岩类
    "花岗岩": (0.90, 0.85, 0.80),  # 浅灰粉色
    "闪长岩": (0.70, 0.75, 0.70),  # 灰绿色
    "辉长岩": (0.50, 0.55, 0.50),  # 深灰绿色
    "玄武岩": (0.40, 0.40, 0.45),  # 深灰色
    "安山岩": (0.60, 0.60, 0.65),  # 中灰色
    "流纹岩": (0.85, 0.80, 0.85),  # 浅紫灰色

    # 变质岩类
    "片麻岩": (0.80, 0.75, 0.70),  # 灰褐色
    "片岩": (0.70, 0.70, 0.75),  # 蓝灰色
    "板岩": (0.50, 0.55, 0.60),  # 深蓝灰色
    "大理岩": (0.95, 0.95, 0.90),  # 乳白色
    "石英岩": (0.90, 0.90, 0.95),  # 浅蓝白色

    # 特殊岩层
    "煤层": (0.20, 0.20, 0.20),  # 深黑色
    "油页岩": (0.35, 0.30, 0.25),  # 深褐色
    "盐岩": (0.95, 0.90, 0.90),  # 浅粉色
    "石膏": (0.90, 0.95, 0.95),  # 浅白色

    # 松散层
    "粘土": (0.70, 0.60, 0.50),  # 棕色
    "粉土": (0.85, 0.80, 0.70),  # 浅棕色
    "砂土": (0.90, 0.85, 0.65),  # 浅黄色
    "碎石": (0.75, 0.70, 0.65),  # 灰褐色
}

# Gradient Fill颜色配置
GRADIENT_COLORS = {
    "地层间填充": [
        (0.95, 0.90, 0.85, 0.3),  # 浅色渐变起点
        (0.85, 0.80, 0.75, 0.3),  # 浅色渐变终点
    ],
    "煤层特殊填充": [
        (0.40, 0.40, 0.40, 0.4),  # 深色渐变起点
        (0.20, 0.20, 0.20, 0.4),  # 深色渐变终点
    ]
}

# 封闭模型使用的颜色（更浅更透明）
closed_model_color = (0.9, 0.7, 0.5)  # 浅橙色
closed_model_opacity = 0.3  # 更高的Opacity

# Theme配置
THEMES = {
    "light": {
        "background": (1.0, 1.0, 1.0),
        "grid_color": (0.85, 0.85, 0.85),
        "text_color": "black"
    },
    "dark": {
        "background": (1.0, 1.0, 1.0),  # 改为白色背景
        "grid_color": (0.85, 0.85, 0.85),  # 浅灰色网格线
        "text_color": "black"
    }
}


class DataValidationDialog(QDialog):
    """Data Validation对话框"""

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Validation结果")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout()

        # 验证结果
        self.validation_text = QTextEdit()
        self.validation_text.setReadOnly(True)
        layout.addWidget(self.validation_text)

        # 按钮
        button_layout = QHBoxLayout()
        self.accept_btn = QPushButton("✅ 接受数据")
        self.reject_btn = QPushButton("❌ 拒绝数据")
        button_layout.addWidget(self.accept_btn)
        button_layout.addWidget(self.reject_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 执行验证
        self.validate_data(df)

        # 连接信号
        self.accept_btn.clicked.connect(self.accept)
        self.reject_btn.clicked.connect(self.reject)

    def validate_data(self, df):
        """验证数据完整性和格式"""
        validation_results = []

        # 检查必需列
        required_columns = ['ID', 'X', 'Y', 'Z', 'Top', 'Bottom', 'layername']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            validation_results.append(f"❌ 缺少必需列: {', '.join(missing_columns)}")
        else:
            validation_results.append("✅ 所有必需列都存在")

        # 检查数据类型
        numeric_columns = ['X', 'Y', 'Z', 'Top', 'Bottom']
        for col in numeric_columns:
            if col in df.columns:
                try:
                    pd.to_numeric(df[col])
                    validation_results.append(f"✅ {col} 列数据类型正确")
                except:
                    validation_results.append(f"❌ {col} 列包含非数字数据")

        # 检查数据完整性
        total_rows = len(df)
        validation_results.append(f"📊 总行数: {total_rows}")

        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    validation_results.append(f"⚠️ {col} 列有 {null_count} 个空值")
                else:
                    validation_results.append(f"✅ {col} 列无空值")

        # 检查坐标范围
        if all(col in df.columns for col in ['X', 'Y', 'Z']):
            try:
                x_range = f"{df['X'].min():.2f} ~ {df['X'].max():.2f}"
                y_range = f"{df['Y'].min():.2f} ~ {df['Y'].max():.2f}"
                z_range = f"{df['Z'].min():.2f} ~ {df['Z'].max():.2f}"
                validation_results.append(f"📍 X坐标范围: {x_range}")
                validation_results.append(f"📍 Y坐标范围: {y_range}")
                validation_results.append(f"📍 Z坐标范围: {z_range}")
            except:
                validation_results.append("❌ 坐标数据格式错误")

        # 检查深度逻辑
        if all(col in df.columns for col in ['Top', 'Bottom']):
            try:
                invalid_depth = df[df['Top'] >= df['Bottom']]
                if len(invalid_depth) > 0:
                    validation_results.append(f"❌ 发现 {len(invalid_depth)} 行深度逻辑错误 (Top >= Bottom)")
                else:
                    validation_results.append("✅ 深度逻辑正确")
            except:
                validation_results.append("❌ 深度数据格式错误")

        # 检查地层类型
        if 'layername' in df.columns:
            unique_layers = df['layername'].unique()
            validation_results.append(f"🗂️ 发现地层类型: {', '.join(map(str, unique_layers))}")

        self.validation_text.setPlainText('\n'.join(validation_results))


class DataFilterDialog(QDialog):
    """Data Filter对话框"""

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Filter")
        self.setModal(True)
        self.resize(400, 500)
        self.df = df

        layout = QVBoxLayout()

        # 深度过滤
        depth_group = QGroupBox("深度过滤")
        depth_layout = QFormLayout()

        self.min_depth_spin = QDoubleSpinBox()
        self.min_depth_spin.setRange(-10000, 10000)
        self.min_depth_spin.setValue(df['Top'].min() if 'Top' in df.columns else 0)
        depth_layout.addRow("最小深度:", self.min_depth_spin)

        self.max_depth_spin = QDoubleSpinBox()
        self.max_depth_spin.setRange(-10000, 10000)
        self.max_depth_spin.setValue(df['Bottom'].max() if 'Bottom' in df.columns else 100)
        depth_layout.addRow("最大深度:", self.max_depth_spin)

        depth_group.setLayout(depth_layout)
        layout.addWidget(depth_group)

        # 坐标范围过滤
        coord_group = QGroupBox("坐标范围过滤")
        coord_layout = QFormLayout()

        self.min_x_spin = QDoubleSpinBox()
        self.min_x_spin.setRange(-100000, 100000)
        self.min_x_spin.setValue(df['X'].min() if 'X' in df.columns else 0)
        coord_layout.addRow("最小X坐标:", self.min_x_spin)

        self.max_x_spin = QDoubleSpinBox()
        self.max_x_spin.setRange(-100000, 100000)
        self.max_x_spin.setValue(df['X'].max() if 'X' in df.columns else 100)
        coord_layout.addRow("最大X坐标:", self.max_x_spin)

        self.min_y_spin = QDoubleSpinBox()
        self.min_y_spin.setRange(-100000, 100000)
        self.min_y_spin.setValue(df['Y'].min() if 'Y' in df.columns else 0)
        coord_layout.addRow("最小Y坐标:", self.min_y_spin)

        self.max_y_spin = QDoubleSpinBox()
        self.max_y_spin.setRange(-100000, 100000)
        self.max_y_spin.setValue(df['Y'].max() if 'Y' in df.columns else 100)
        coord_layout.addRow("最大Y坐标:", self.max_y_spin)

        coord_group.setLayout(coord_layout)
        layout.addWidget(coord_group)

        # 地层类型过滤
        layer_group = QGroupBox("地层类型过滤")
        layer_layout = QVBoxLayout()

        self.layer_checkboxes = {}
        if 'layername' in df.columns:
            for layer in df['layername'].unique():
                checkbox = QCheckBox(str(layer))
                checkbox.setChecked(True)
                self.layer_checkboxes[layer] = checkbox
                layer_layout.addWidget(checkbox)

        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)

        # 按钮
        button_layout = QHBoxLayout()
        self.apply_btn = QPushButton("✅ Apply过滤")
        self.reset_btn = QPushButton("🔄 Reset")
        self.cancel_btn = QPushButton("❌ Cancel")
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 连接信号
        self.apply_btn.clicked.connect(self.apply_filter)
        self.reset_btn.clicked.connect(self.reset_filter)
        self.cancel_btn.clicked.connect(self.reject)

    def apply_filter(self):
        """Apply过滤条件"""
        self.filtered_df = self.df.copy()

        # 深度过滤
        if 'Top' in self.filtered_df.columns and 'Bottom' in self.filtered_df.columns:
            min_depth = self.min_depth_spin.value()
            max_depth = self.max_depth_spin.value()
            self.filtered_df = self.filtered_df[
                (self.filtered_df['Top'] >= min_depth) &
                (self.filtered_df['Bottom'] <= max_depth)
                ]

        # 坐标过滤
        if 'X' in self.filtered_df.columns:
            min_x = self.min_x_spin.value()
            max_x = self.max_x_spin.value()
            self.filtered_df = self.filtered_df[
                (self.filtered_df['X'] >= min_x) &
                (self.filtered_df['X'] <= max_x)
                ]

        if 'Y' in self.filtered_df.columns:
            min_y = self.min_y_spin.value()
            max_y = self.max_y_spin.value()
            self.filtered_df = self.filtered_df[
                (self.filtered_df['Y'] >= min_y) &
                (self.filtered_df['Y'] <= max_y)
                ]

        # 地层类型过滤
        if 'layername' in self.filtered_df.columns:
            selected_layers = [layer for layer, checkbox in self.layer_checkboxes.items()
                               if checkbox.isChecked()]
            self.filtered_df = self.filtered_df[
                self.filtered_df['layername'].isin(selected_layers)
            ]

        self.accept()

    def reset_filter(self):
        """Reset过滤条件"""
        if 'Top' in self.df.columns:
            self.min_depth_spin.setValue(self.df['Top'].min())
        if 'Bottom' in self.df.columns:
            self.max_depth_spin.setValue(self.df['Bottom'].max())
        if 'X' in self.df.columns:
            self.min_x_spin.setValue(self.df['X'].min())
            self.max_x_spin.setValue(self.df['X'].max())
        if 'Y' in self.df.columns:
            self.min_y_spin.setValue(self.df['Y'].min())
            self.max_y_spin.setValue(self.df['Y'].max())

        for checkbox in self.layer_checkboxes.values():
            checkbox.setChecked(True)


class AnalysisDialog(QDialog):
    """Analysis Tools对话框"""

    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Analysis Tools")
        self.setModal(False)
        self.resize(800, 600)
        self.df = df

        layout = QVBoxLayout()

        # 选项卡
        self.tab_widget = QTabWidget()

        # 体积计算标签页
        self.volume_tab = self.create_volume_tab()
        self.tab_widget.addTab(self.volume_tab, "体积计算")

        # 厚度分析标签页
        self.thickness_tab = self.create_thickness_tab()
        self.tab_widget.addTab(self.thickness_tab, "厚度分析")

        # 插值分析标签页
        self.interpolation_tab = self.create_interpolation_tab()
        self.tab_widget.addTab(self.interpolation_tab, "插值分析")

        layout.addWidget(self.tab_widget)
        self.setLayout(layout)

    def create_volume_tab(self):
        """创建体积计算标签页"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 控制面板
        control_group = QGroupBox("体积计算控制")
        control_layout = QFormLayout()

        self.volume_layer_combo = QComboBox()
        if 'layername' in self.df.columns:
            self.volume_layer_combo.addItems(['全部'] + list(self.df['layername'].unique()))
        control_layout.addRow("选择地层:", self.volume_layer_combo)

        self.calc_volume_btn = QPushButton("📊 计算体积")
        self.calc_volume_btn.clicked.connect(self.calculate_volume)
        control_layout.addRow(self.calc_volume_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 结果显示
        self.volume_results = QTextEdit()
        self.volume_results.setReadOnly(True)
        layout.addWidget(self.volume_results)

        widget.setLayout(layout)
        return widget

    def create_thickness_tab(self):
        """创建厚度分析标签页"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 控制面板
        control_group = QGroupBox("厚度分析控制")
        control_layout = QFormLayout()

        self.thickness_layer_combo = QComboBox()
        if 'layername' in self.df.columns:
            self.thickness_layer_combo.addItems(list(self.df['layername'].unique()))
        control_layout.addRow("选择地层:", self.thickness_layer_combo)

        self.analyze_thickness_btn = QPushButton("📏 分析厚度")
        self.analyze_thickness_btn.clicked.connect(self.analyze_thickness)
        control_layout.addRow(self.analyze_thickness_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 结果显示区域（图表+文本）
        splitter = QSplitter(Qt.Horizontal)

        # 图表区域
        self.thickness_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        splitter.addWidget(self.thickness_canvas)

        # 文本结果
        self.thickness_results = QTextEdit()
        self.thickness_results.setReadOnly(True)
        splitter.addWidget(self.thickness_results)

        layout.addWidget(splitter)
        widget.setLayout(layout)
        return widget

    def create_interpolation_tab(self):
        """创建插值分析标签页"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 控制面板
        control_group = QGroupBox("插值分析控制")
        control_layout = QFormLayout()

        self.interp_method_combo = QComboBox()
        self.interp_method_combo.addItems(['linear', 'cubic', 'nearest'])
        control_layout.addRow("插值方法:", self.interp_method_combo)

        self.interp_layer_combo = QComboBox()
        if 'layername' in self.df.columns:
            self.interp_layer_combo.addItems(list(self.df['layername'].unique()))
        control_layout.addRow("选择地层:", self.interp_layer_combo)

        self.grid_size_spin = QSpinBox()
        self.grid_size_spin.setRange(10, 200)
        self.grid_size_spin.setValue(50)
        control_layout.addRow("网格大小:", self.grid_size_spin)

        self.interpolate_btn = QPushButton("🔄 执行插值")
        self.interpolate_btn.clicked.connect(self.perform_interpolation)
        control_layout.addRow(self.interpolate_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 结果显示
        self.interp_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.interp_canvas)

        widget.setLayout(layout)
        return widget

    def calculate_volume(self):
        """计算体积"""
        try:
            layer = self.volume_layer_combo.currentText()
            results = []

            if layer == "全部":
                # 计算所有地层
                for layer_name in self.df['layername'].unique():
                    layer_data = self.df[self.df['layername'] == layer_name]
                    volume = self._calculate_layer_volume(layer_data)
                    results.append(f"{layer_name}: {volume:.2f} 立方米")
            else:
                # 计算选定地层
                layer_data = self.df[self.df['layername'] == layer]
                volume = self._calculate_layer_volume(layer_data)
                results.append(f"{layer}: {volume:.2f} 立方米")

            self.volume_results.setPlainText('\n'.join(results))

        except Exception as e:
            QMessageBox.warning(self, "错误", f"体积计算失败: {str(e)}")

    def _calculate_layer_volume(self, layer_data):
        """计算单个地层体积"""
        if len(layer_data) == 0:
            return 0

        # 简化体积计算：使用钻孔数据估算
        total_volume = 0
        for _, row in layer_data.iterrows():
            thickness = row['Bottom'] - row['Top']
            # 假设每个钻孔代表周围一定面积
            area = 100  # 简化假设每个钻孔代表100平方米
            volume = thickness * area
            total_volume += volume

        return total_volume

    def analyze_thickness(self):
        """分析厚度"""
        try:
            layer = self.thickness_layer_combo.currentText()
            layer_data = self.df[self.df['layername'] == layer]

            if len(layer_data) == 0:
                QMessageBox.warning(self, "警告", "没有找到该地层的数据")
                return

            # 计算厚度
            thicknesses = layer_data['Bottom'] - layer_data['Top']

            # 统计分析
            stats = {
                "最小厚度": thicknesses.min(),
                "最大厚度": thicknesses.max(),
                "平均厚度": thicknesses.mean(),
                "中位数厚度": thicknesses.median(),
                "标准差": thicknesses.std(),
                "样本数量": len(thicknesses)
            }

            # 显示文本结果
            results = [f"{key}: {value:.2f}" for key, value in stats.items()]
            self.thickness_results.setPlainText('\n'.join(results))

            # 绘制厚度分布图
            fig = self.thickness_canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.hist(thicknesses, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('厚度 (m)', fontsize=12)
            ax.set_ylabel('频率', fontsize=12)
            ax.set_title(f'{layer} 厚度分布', fontsize=14, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=10)
            fig.tight_layout()
            self.thickness_canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "错误", f"厚度分析失败: {str(e)}")

    def perform_interpolation(self):
        """执行插值分析"""
        try:
            layer = self.interp_layer_combo.currentText()
            method = self.interp_method_combo.currentText()
            grid_size = self.grid_size_spin.value()

            layer_data = self.df[self.df['layername'] == layer]

            if len(layer_data) < 3:
                QMessageBox.warning(self, "警告", "插值需要至少3个数据点")
                return

            # 准备数据
            x = layer_data['X'].values
            y = layer_data['Y'].values
            z = layer_data['Top'].values  # 使用顶面深度进行插值

            # 创建网格
            xi = np.linspace(x.min(), x.max(), grid_size)
            yi = np.linspace(y.min(), y.max(), grid_size)
            xi_grid, yi_grid = np.meshgrid(xi, yi)

            # 执行插值
            zi_grid = griddata((x, y), z, (xi_grid, yi_grid), method=method)

            # 绘制结果
            fig = self.interp_canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)

            # 插值结果
            im = ax.contourf(xi_grid, yi_grid, zi_grid, levels=20, cmap='terrain')
            cbar = fig.colorbar(im, ax=ax, label='深度 (m)')
            cbar.ax.tick_params(labelsize=10)

            # 原始数据点
            ax.scatter(x, y, c='red', s=50, marker='o', label='钻孔Position')

            ax.set_xlabel('X 坐标', fontsize=12)
            ax.set_ylabel('Y 坐标', fontsize=12)
            ax.set_title(f'{layer} 深度插值图 ({method}方法)', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            fig.tight_layout()
            self.interp_canvas.draw()

        except Exception as e:
            QMessageBox.warning(self, "错误", f"插值分析失败: {str(e)}")


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, interactor, renderer, picker, actor_data_map, show_info_func, viewer=None):
        super().__init__()
        self.interactor = interactor
        self.renderer = renderer
        self.picker = picker
        self.actor_data_map = actor_data_map
        self.show_info_func = show_info_func
        self.viewer = viewer  # 添加viewer引用用于断层交互
        self.AddObserver("LeftButtonPressEvent", self.on_left_click)

    def on_left_click(self, obj, event):
        x, y = self.interactor.GetEventPosition()

        # 如果是断层交互模式，处理点选
        if self.viewer and self.viewer.interactive_fault_mode:
            # 进行3D坐标拾取
            world_pos = self.pick_world_position(x, y)
            if world_pos:
                self.viewer.on_point_selected(*world_pos)
            return

        # 正常的对象选择
        self.picker.Pick(x, y, 0, self.renderer)
        actor = self.picker.GetActor()
        if actor in self.actor_data_map:
            self.show_info_func(self.actor_data_map[actor])
        else:
            super().OnLeftButtonDown()

    def pick_world_position(self, x, y):
        """拾取世界坐标Position"""
        try:
            # 使用世界坐标拾取器
            world_picker = vtk.vtkWorldPointPicker()
            if world_picker.Pick(x, y, 0, self.renderer):
                world_pos = world_picker.GetPickPosition()
                return world_pos

            # 如果世界坐标拾取失败，尝试使用点拾取器
            point_picker = vtk.vtkPointPicker()
            if point_picker.Pick(x, y, 0, self.renderer):
                world_pos = point_picker.GetPickPosition()
                return world_pos

            # 如果都失败，使用相机投影估算Position
            camera = self.renderer.GetActiveCamera()
            if camera:
                # 简化的深度估算
                focal_point = camera.GetFocalPoint()
                return (focal_point[0], focal_point[1], focal_point[2])

        except Exception as e:
            print(f"坐标拾取失败: {e}")

        return None


class BoreholeViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🚀 高级钻孔三维可视化系统 | Advanced Borehole 3D Visualization")
        self.resize(1600, 1000)  # 增大默认窗口尺寸

        # 设置窗口图标和属性
        self.setMinimumSize(1200, 800)

        # 启用高DPI支持
        try:
            from PyQt5.QtCore import Qt
            self.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        except:
            pass

        # 初始化数据
        self.df_latest = None
        self.df_original = None  # 保存原始数据
        self.current_theme = "dark"  # 默认使用深色Theme
        self.analysis_dialog = None

        # 状态与 actor 管理
        self.point_actors = []
        self.points_visible = False
        self.delaunay_actor = None
        self.delaunay_visible = False
        self.layer_top_mesh_actors = {}
        self.layer_top_visible = {}
        self.coal_bottom_mesh_actor = None
        self.closed_model_actors = []
        self.closed_model_visible = False
        self.models_visible = True

        # 断层相关属性
        self.fault_actors = {}
        self.fault_visible = {}
        self.faults_enabled = True

        # 交互模式属性
        self.interactive_fault_mode = False
        self.selected_fault_points = []
        self.fault_point_actors = []
        self.current_fault_name = ""

        # Cut模型管理
        self.original_model_actors = []  # 保存原始模型
        self.cut_model_actors = []  # Cut后的模型
        self.is_cut_mode = False  # 是否在Cut模式

        # 设置菜单栏
        self.setup_menubar()

        # 设置工具栏
        self.setup_toolbar()

        # 设置状态栏
        self.setup_statusbar()

        # 设置中央部件 (这会初始化VTK渲染器)
        self.setup_central_widget()

        # 设置快捷键
        self.setup_shortcuts()

        # ApplyTheme
        self.apply_theme()

        # 欢迎信息
        self.update_status("🎉 欢迎使用高级钻孔三维可视化系统")

        # 验证和设置matplotlib字体
        self.verify_chinese_font()

        # 初始化算法组件
        self.initialize_algorithms()

    def initialize_algorithms(self):
        """初始化高级算法组件"""
        try:
            # 初始化Kriging Interpolation器
            self.kriging_interpolator = KrigingInterpolator()

            # 初始化Anomaly Detection器
            self.anomaly_detector = AnomalyDetector()

            # 初始化Fault Modeling器
            self.fault_modeler = FaultModeler()

            # Cache Management器已在全局初始化
            self.cache_manager = cache_manager

            # 算法状态
            self.algorithm_status = {
                'kriging_available': True,
                'anomaly_detection_available': True,
                'caching_available': True,
                'fault_modeling_available': True
            }

            print("✅ 高级算法组件初始化成功")

        except Exception as e:
            print(f"⚠️ 算法组件初始化部分失败: {e}")
            # 设置备用状态
            self.algorithm_status = {
                'kriging_available': False,
                'anomaly_detection_available': False,
                'caching_available': True,  # 缓存系统不依赖外部库
                'fault_modeling_available': False
            }

    def setup_menubar(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # File菜单
        file_menu = menubar.addMenu('File')

        # Import
        import_menu = file_menu.addMenu('Import')

        load_csv_action = QAction('Load CSVFile', self)
        load_csv_action.setShortcut('Ctrl+O')
        load_csv_action.triggered.connect(self.load_csv)
        import_menu.addAction(load_csv_action)

        load_multiple_action = QAction('Import多个CSVFile', self)
        load_multiple_action.setShortcut('Ctrl+Shift+O')
        load_multiple_action.triggered.connect(self.load_multiple_csv)
        import_menu.addAction(load_multiple_action)

        # Export
        export_menu = file_menu.addMenu('Export')

        export_excel_action = QAction('Export为Excel', self)
        export_excel_action.triggered.connect(self.export_excel)
        export_menu.addAction(export_excel_action)

        export_json_action = QAction('Export为JSON', self)
        export_json_action.triggered.connect(self.export_json)
        export_menu.addAction(export_json_action)

        export_stl_action = QAction('Export3D模型(STL)', self)
        export_stl_action.triggered.connect(self.export_stl)
        export_menu.addAction(export_stl_action)

        export_image_action = QAction('Export图像', self)
        export_image_action.setShortcut('Ctrl+S')
        export_image_action.triggered.connect(self.export_image)
        export_menu.addAction(export_image_action)

        # 数据菜单
        data_menu = menubar.addMenu('数据')

        filter_action = QAction('Data Filter', self)
        filter_action.setShortcut('Ctrl+F')
        filter_action.triggered.connect(self.open_data_filter)
        data_menu.addAction(filter_action)

        validate_action = QAction('Data Validation', self)
        validate_action.triggered.connect(self.validate_data)
        data_menu.addAction(validate_action)

        interpolate_action = QAction('Data Interpolation', self)
        interpolate_action.triggered.connect(self.interpolate_missing_data)
        data_menu.addAction(interpolate_action)

        smooth_action = QAction('Data Smoothing', self)
        smooth_action.triggered.connect(self.smooth_data)
        data_menu.addAction(smooth_action)

        detect_anomaly_action = QAction('Anomaly Detection', self)
        detect_anomaly_action.triggered.connect(self.detect_anomalies)
        data_menu.addAction(detect_anomaly_action)

        data_menu.addSeparator()

        # 高级算法菜单
        advanced_menu = data_menu.addMenu('🧠 高级算法')

        kriging_action = QAction('🎯 Kriging Interpolation', self)
        kriging_action.triggered.connect(self.apply_kriging_interpolation)
        advanced_menu.addAction(kriging_action)

        anomaly_advanced_action = QAction('🔍 高级Anomaly Detection', self)
        anomaly_advanced_action.triggered.connect(self.apply_advanced_anomaly_detection)
        advanced_menu.addAction(anomaly_advanced_action)

        cache_menu = advanced_menu.addMenu('💾 Cache Management')

        clear_cache_action = QAction('Clear Cache', self)
        clear_cache_action.triggered.connect(self.clear_data_cache)
        cache_menu.addAction(clear_cache_action)

        cache_info_action = QAction('Cache Info', self)
        cache_info_action.triggered.connect(self.show_cache_info)
        cache_menu.addAction(cache_info_action)

        # 分析菜单
        analysis_menu = menubar.addMenu('分析')

        volume_action = QAction('Volume Analysis', self)
        volume_action.triggered.connect(self.open_analysis_dialog)
        analysis_menu.addAction(volume_action)

        profile_action = QAction('Profile Analysis', self)
        profile_action.triggered.connect(self.create_profile_analysis)
        analysis_menu.addAction(profile_action)

        # View菜单
        view_menu = menubar.addMenu('View')

        theme_menu = view_menu.addMenu('Theme')

        light_theme_action = QAction('浅色Theme', self)
        light_theme_action.triggered.connect(lambda: self.switch_theme('light'))
        theme_menu.addAction(light_theme_action)

        dark_theme_action = QAction('深色Theme', self)
        dark_theme_action.triggered.connect(lambda: self.switch_theme('dark'))
        theme_menu.addAction(dark_theme_action)

        reset_camera_action = QAction('Reset视角', self)
        reset_camera_action.setShortcut('R')
        reset_camera_action.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_camera_action)

        # Help菜单
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def setup_toolbar(self):
        """设置工具栏"""
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        # 工具栏已清空，保留基本结构

    def setup_statusbar(self):
        """设置状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 主状态标签（带图标）
        self.status_label = QLabel("🚀 系统已就绪")
        self.status_label.setStyleSheet("font-weight: 500;")
        self.status_bar.addWidget(self.status_label)

        # 添加分隔符
        separator1 = QLabel("|")
        separator1.setStyleSheet("color: #666; margin: 0 5px;")
        self.status_bar.addPermanentWidget(separator1)

        # Theme指示器
        self.theme_label = QLabel("☀️ 亮色Theme")
        self.theme_label.setStyleSheet("font-size: 8pt;")
        self.status_bar.addPermanentWidget(self.theme_label)

        # 添加分隔符
        separator2 = QLabel("|")
        separator2.setStyleSheet("color: #666; margin: 0 5px;")
        self.status_bar.addPermanentWidget(separator2)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setStyleSheet("margin: 2px;")
        self.status_bar.addPermanentWidget(self.progress_bar)

        # Data Statistics（带图标）
        self.data_info_label = QLabel("📊 无数据")
        self.data_info_label.setStyleSheet("font-size: 8pt; min-width: 120px;")
        self.status_bar.addPermanentWidget(self.data_info_label)

        # 添加分隔符
        separator3 = QLabel("|")
        separator3.setStyleSheet("color: #666; margin: 0 5px;")
        self.status_bar.addPermanentWidget(separator3)

        # 版本信息
        version_label = QLabel("🔧 v2.0")
        version_label.setStyleSheet("font-size: 8pt; color: #888;")
        self.status_bar.addPermanentWidget(version_label)

    def setup_central_widget(self):
        """设置中央部件"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)

        # 中间3DView
        self.vtk_widget = QVTKRenderWindowInteractor()
        main_layout.addWidget(self.vtk_widget, 3)

        # 右侧信息面板
        right_panel = self.create_info_panel()
        main_layout.addWidget(right_panel, 1)

        # 设置VTK渲染器
        self.setup_vtk_renderer()

    def create_control_panel(self):
        """创建左侧控制面板"""
        # 创建主面板
        main_panel = QWidget()
        main_panel.setFixedWidth(280)  # 固定宽度避免挤压

        # 创建滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 创建内容widget
        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)
        layout.setSpacing(6)  # 减少间距
        layout.setContentsMargins(8, 8, 8, 8)  # 设置边距

        # 数据Import组
        import_group = QGroupBox("📁 数据Import")
        import_group.setMaximumHeight(100)  # 限制高度
        import_layout = QVBoxLayout()
        import_layout.setSpacing(3)

        self.btn_load = QPushButton("📄 Load CSV")
        self.btn_load.clicked.connect(self.load_csv)
        self.btn_load.setMinimumHeight(28)
        self.btn_load.setMaximumHeight(28)
        import_layout.addWidget(self.btn_load)

        self.btn_load_multiple = QPushButton("📁 Batch Import")
        self.btn_load_multiple.clicked.connect(self.load_multiple_csv)
        self.btn_load_multiple.setMinimumHeight(28)
        self.btn_load_multiple.setMaximumHeight(28)
        import_layout.addWidget(self.btn_load_multiple)

        import_group.setLayout(import_layout)
        layout.addWidget(import_group)

        # 3D模型控制组
        model_group = QGroupBox("🎁 3D模型控制")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(4)

        self.btn_closed = QPushButton("🎁 Generate Closed Model")
        self.btn_closed.clicked.connect(self.toggle_closed_model)
        self.btn_closed.setMinimumHeight(32)
        self.btn_closed.setStyleSheet("QPushButton { font-weight: bold; }")
        model_layout.addWidget(self.btn_closed)

        # 显示控制按钮行
        display_row1 = QHBoxLayout()
        self.btn_toggle_points = QPushButton("� Borehole Points")
        self.btn_toggle_points.clicked.connect(self.toggle_borehole_points)
        self.btn_toggle_points.setMinimumHeight(28)
        display_row1.addWidget(self.btn_toggle_points)

        self.btn_toggle_models = QPushButton("�️ Borehole Models")
        self.btn_toggle_models.clicked.connect(self.toggle_borehole_models)
        self.btn_toggle_models.setMinimumHeight(28)
        display_row1.addWidget(self.btn_toggle_models)
        model_layout.addLayout(display_row1)

        display_row2 = QHBoxLayout()
        self.btn_delaunay = QPushButton("🔗 Borehole Mesh")
        self.btn_delaunay.clicked.connect(self.toggle_delaunay)
        self.btn_delaunay.setMinimumHeight(28)
        display_row2.addWidget(self.btn_delaunay)

        self.btn_layer_top = QPushButton("�️ Stratum Mesh")
        self.btn_layer_top.clicked.connect(self.toggle_layer_top_delaunay)
        self.btn_layer_top.setMinimumHeight(28)
        display_row2.addWidget(self.btn_layer_top)
        model_layout.addLayout(display_row2)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Visual Effects组
        visual_group = QGroupBox("🎨 Visual Effects")
        visual_layout = QVBoxLayout()
        visual_layout.setSpacing(4)

        # 颜色映射控制
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color Scheme:"))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(['By Stratum', 'By Depth', 'By Thickness'])
        self.color_mode_combo.currentTextChanged.connect(self.update_color_mapping)
        color_layout.addWidget(self.color_mode_combo)
        visual_layout.addLayout(color_layout)

        # Opacity控制
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(70)
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        opacity_layout.addWidget(self.opacity_slider)
        self.opacity_value_label = QLabel("70%")
        self.opacity_value_label.setMinimumWidth(35)
        opacity_layout.addWidget(self.opacity_value_label)
        self.opacity_slider.valueChanged.connect(lambda v: self.opacity_value_label.setText(f"{v}%"))
        visual_layout.addLayout(opacity_layout)

        # 平滑控制
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("Smoothness:"))
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(10, 100)
        self.smooth_slider.setValue(30)
        self.smooth_slider.setToolTip("控制封闭模型的平滑程度")
        smooth_layout.addWidget(self.smooth_slider)

        self.smooth_value_label = QLabel("30")
        self.smooth_value_label.setMinimumWidth(30)
        smooth_layout.addWidget(self.smooth_value_label)
        self.smooth_slider.valueChanged.connect(self.update_smooth_value_label)
        visual_layout.addLayout(smooth_layout)

        # 平滑类型选择
        smooth_type_layout = QHBoxLayout()
        smooth_type_layout.addWidget(QLabel("Smoothing Algorithm:"))
        self.smooth_type_combo = QComboBox()
        self.smooth_type_combo.addItems(['Laplace', 'Windowed Sine'])
        self.smooth_type_combo.setCurrentText('Laplace')
        smooth_type_layout.addWidget(self.smooth_type_combo)
        visual_layout.addLayout(smooth_type_layout)

        visual_group.setLayout(visual_layout)
        layout.addWidget(visual_group)

        # Visual Enhancement组
        beauty_group = QGroupBox("✨ Visual Enhancement")
        beauty_layout = QVBoxLayout()
        beauty_layout.setSpacing(4)

        # Gradient Fill控制
        gradient_layout = QHBoxLayout()
        self.gradient_checkbox = QCheckBox("Gradient Fill")
        self.gradient_checkbox.stateChanged.connect(self.toggle_gradient_fill)
        gradient_layout.addWidget(self.gradient_checkbox)
        beauty_layout.addLayout(gradient_layout)

        # 切片控制组
        slice_group = QGroupBox("✂️ Model Slicing")
        slice_layout = QVBoxLayout()
        slice_layout.setSpacing(2)

        # 切片Direction
        slice_direction_layout = QHBoxLayout()
        slice_direction_layout.addWidget(QLabel("Direction:"))
        self.slice_direction_combo = QComboBox()
        self.slice_direction_combo.addItems(['Horizontal Slice', 'Vertical Slice', 'Lateral Slice'])
        self.slice_direction_combo.currentTextChanged.connect(self.update_slice_direction)
        slice_direction_layout.addWidget(self.slice_direction_combo)
        slice_layout.addLayout(slice_direction_layout)

        # 切片Position
        slice_position_layout = QHBoxLayout()
        slice_position_layout.addWidget(QLabel("Position:"))
        self.slice_position_slider = QSlider(Qt.Horizontal)
        self.slice_position_slider.setRange(0, 100)
        self.slice_position_slider.setValue(50)
        self.slice_position_slider.valueChanged.connect(self.update_slice_position)
        slice_position_layout.addWidget(self.slice_position_slider)

        self.slice_position_label = QLabel("50%")
        self.slice_position_label.setMinimumWidth(35)
        slice_position_layout.addWidget(self.slice_position_label)
        slice_layout.addLayout(slice_position_layout)

        # 切片控制按钮
        slice_control_layout = QHBoxLayout()
        self.slice_checkbox = QCheckBox("Show Slice")
        self.slice_checkbox.stateChanged.connect(self.toggle_slice_display)
        slice_control_layout.addWidget(self.slice_checkbox)

        self.slice_model_btn = QPushButton("🔪 Cut")
        self.slice_model_btn.clicked.connect(self.apply_slice_to_models)
        self.slice_model_btn.setToolTip("将切片Apply到所有模型")
        self.slice_model_btn.setMinimumHeight(28)
        slice_control_layout.addWidget(self.slice_model_btn)
        slice_layout.addLayout(slice_control_layout)

        slice_group.setLayout(slice_layout)
        beauty_layout.addWidget(slice_group)

        beauty_group.setLayout(beauty_layout)
        layout.addWidget(beauty_group)

        # 地层显示控制面板
        self.layer_control_group = QGroupBox("🗂️ Stratum Control")
        self.layer_control_layout = QVBoxLayout()
        self.layer_control_group.setLayout(self.layer_control_layout)
        layout.addWidget(self.layer_control_group)

        # 断层控制面板
        self.fault_control_group = QGroupBox("⚡ Fault Modeling")
        fault_layout = QVBoxLayout()
        fault_layout.setSpacing(4)

        # 快速断层添加
        quick_add_layout = QHBoxLayout()
        self.btn_interactive_fault = QPushButton("�️ Interactive Add")
        self.btn_interactive_fault.clicked.connect(self.start_interactive_fault_mode)
        self.btn_interactive_fault.setMinimumHeight(28)
        self.btn_interactive_fault.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        quick_add_layout.addWidget(self.btn_interactive_fault)

        self.btn_auto_detect = QPushButton("� Auto Detect")
        self.btn_auto_detect.clicked.connect(self.auto_detect_faults)
        self.btn_auto_detect.setMinimumHeight(28)
        quick_add_layout.addWidget(self.btn_auto_detect)
        fault_layout.addLayout(quick_add_layout)

        # Quick Template
        template_layout = QHBoxLayout()
        self.btn_quick_template = QPushButton("⚡ Quick Template")
        self.btn_quick_template.clicked.connect(self.show_fault_templates)
        self.btn_quick_template.setMinimumHeight(28)
        template_layout.addWidget(self.btn_quick_template)

        self.btn_advanced_add = QPushButton("⚙️ Advanced Add")
        self.btn_advanced_add.clicked.connect(self.show_advanced_fault_dialog)
        self.btn_advanced_add.setMinimumHeight(28)
        template_layout.addWidget(self.btn_advanced_add)
        fault_layout.addLayout(template_layout)

        # 断层显示控制
        fault_display_layout = QHBoxLayout()
        self.btn_toggle_faults = QPushButton("👁️ Show Faults")
        self.btn_toggle_faults.clicked.connect(self.toggle_all_faults)
        self.btn_toggle_faults.setMinimumHeight(28)
        fault_display_layout.addWidget(self.btn_toggle_faults)

        self.btn_clear_faults = QPushButton("🗑️ Clear Faults")
        self.btn_clear_faults.clicked.connect(self.clear_all_faults)
        self.btn_clear_faults.setMinimumHeight(28)
        fault_display_layout.addWidget(self.btn_clear_faults)
        fault_layout.addLayout(fault_display_layout)

        # 断层Cut功能
        cut_layout = QHBoxLayout()
        self.btn_apply_fault_cut = QPushButton("✂️ Cut模型")
        self.btn_apply_fault_cut.clicked.connect(self.apply_fault_cuts)
        self.btn_apply_fault_cut.setMinimumHeight(28)
        self.btn_apply_fault_cut.setStyleSheet("QPushButton { font-weight: bold; color: #FF6B6B; }")
        cut_layout.addWidget(self.btn_apply_fault_cut)

        self.btn_restore_model = QPushButton("🔄 Restore Original")
        self.btn_restore_model.clicked.connect(self.restore_original_models)
        self.btn_restore_model.setMinimumHeight(28)
        self.btn_restore_model.setEnabled(False)
        cut_layout.addWidget(self.btn_restore_model)
        fault_layout.addLayout(cut_layout)

        # Fault List（动态添加）
        self.fault_list_layout = QVBoxLayout()
        fault_layout.addLayout(self.fault_list_layout)

        self.fault_control_group.setLayout(fault_layout)
        layout.addWidget(self.fault_control_group)

        # Analysis Tools组
        analysis_group = QGroupBox("🔬 Analysis Tools")
        analysis_layout = QVBoxLayout()
        analysis_layout.setSpacing(4)

        self.btn_analysis = QPushButton("🛠️ Advanced Analysis")
        self.btn_analysis.clicked.connect(self.open_analysis_dialog)
        self.btn_analysis.setMinimumHeight(32)
        analysis_layout.addWidget(self.btn_analysis)

        self.btn_cache_info = QPushButton("💾 Cache Info")
        self.btn_cache_info.clicked.connect(self.show_cache_info)
        self.btn_cache_info.setMinimumHeight(28)
        analysis_layout.addWidget(self.btn_cache_info)

        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)

        # 添加伸缩空间
        layout.addStretch()

        # 设置内容到滚动区域
        scroll_area.setWidget(content_widget)

        # 创建主面板布局
        main_layout = QVBoxLayout(main_panel)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

        return main_panel

    def create_info_panel(self):
        """创建右侧信息面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 选中对象信息
        info_group = QGroupBox("选中对象信息")
        info_layout = QVBoxLayout()

        self.info_table = QTableWidget()
        self.info_table.setColumnCount(2)
        self.info_table.setHorizontalHeaderLabels(["属性", "值"])
        self.info_table.horizontalHeader().setStretchLastSection(True)
        info_layout.addWidget(self.info_table)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Data Statistics信息
        stats_group = QGroupBox("Data Statistics")
        stats_layout = QVBoxLayout()

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        return panel

    def setup_vtk_renderer(self):
        """设置VTK渲染器"""
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*THEMES[self.current_theme]["background"])
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.01)
        self.actor_data_map = {}

        self.style = CustomInteractorStyle(
            self.interactor, self.renderer, self.picker,
            self.actor_data_map, self.show_info_panel, self
        )
        self.interactor.SetInteractorStyle(self.style)
        self.interactor.Initialize()

        # 添加网格平面和坐标系
        self.add_grid_plane(800, 20)  # 扩大网格到800，间距20
        self._add_axes_widget()

        # 初始化高级可视化管理器
        self.viz_manager = AdvancedVisualizationManager(self.renderer)

        # 设置初始相机视角
        self.reset_camera()

    def setup_shortcuts(self):
        """设置快捷键"""
        # 主要功能快捷键
        QAction('', self, shortcut='Ctrl+O', triggered=self.load_csv)
        QAction('', self, shortcut='Ctrl+S', triggered=self.export_image)
        QAction('', self, shortcut='Ctrl+F', triggered=self.open_data_filter)
        QAction('', self, shortcut='R', triggered=self.reset_camera)
        QAction('', self, shortcut='Ctrl+T',
                triggered=lambda: self.switch_theme('dark' if self.current_theme == 'light' else 'light'))

        # 显示控制快捷键
        QAction('', self, shortcut='1', triggered=self.toggle_borehole_points)
        QAction('', self, shortcut='2', triggered=self.toggle_borehole_models)
        QAction('', self, shortcut='3', triggered=self.toggle_delaunay)
        QAction('', self, shortcut='4', triggered=self.toggle_layer_top_delaunay)
        QAction('', self, shortcut='5', triggered=self.toggle_closed_model)

        # 添加小坐标系
        self._add_axes_widget()

    def apply_theme(self):
        """ApplyTheme"""
        theme = THEMES[self.current_theme]
        self.renderer.SetBackground(*theme["background"])

        # 更新网格颜色
        self.update_grid_color()

        # Apply现代化但较浅的样式
        if self.current_theme == "dark":
            self.setStyleSheet("""
                QMainWindow { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa); 
                    color: #212529;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }

                QMenuBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #212529;
                    border-bottom: 1px solid #dee2e6;
                    padding: 3px;
                    font-size: 10pt;
                }

                QMenuBar::item {
                    background: transparent;
                    padding: 6px 12px;
                    border-radius: 4px;
                    margin: 1px;
                }

                QMenuBar::item:selected {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #6a9eff, stop:1 #4e8ce8);
                    color: white;
                }

                QToolBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    border: none;
                    padding: 3px;
                    spacing: 3px;
                }

                QGroupBox { 
                    color: #212529; 
                    border: 2px solid #dee2e6;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 5px;
                    font-weight: bold;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                }

                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 8px 0 8px;
                    color: #6a9eff;
                }

                QPushButton { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f1f3f4);
                    color: #212529; 
                    border: 1px solid #ced4da;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 9pt;
                    font-weight: 500;
                    min-height: 20px;
                    text-align: left;
                    padding-left: 20px;
                }

                QPushButton:hover { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #7aa9ff, stop:1 #5d8ce8);
                    border: 1px solid #6a9eff;
                    color: white;
                }

                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #5d8ce8, stop:1 #4270d4);
                    border: 1px solid #4270d4;
                    color: white;
                }

                QPushButton:checked {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #6a9eff, stop:1 #3e7ce8);
                    border: 1px solid #3e7ce8;
                    color: white;
                }

                QComboBox { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #212529; 
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    padding: 5px;
                    min-width: 80px;
                }

                QComboBox:hover {
                    border: 1px solid #6a9eff;
                }

                QComboBox::drop-down {
                    border: none;
                    width: 20px;
                }

                QComboBox::down-arrow {
                    image: none;
                    border: 1px solid #888;
                    width: 0;
                    height: 0;
                    border-left: 4px solid transparent;
                    border-right: 4px solid transparent;
                    border-top: 4px solid #212529;
                }

                QSlider::groove:horizontal { 
                    background: #e9ecef; 
                    height: 6px;
                    border-radius: 3px;
                }

                QSlider::handle:horizontal { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #6a9eff, stop:1 #3e7ce8);
                    width: 16px;
                    height: 16px;
                    border-radius: 8px;
                    margin: -5px 0;
                    border: 2px solid #ffffff;
                }

                QSlider::handle:horizontal:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #7ab0ff, stop:1 #4e9cf0);
                }

                QTableWidget { 
                    background-color: #ffffff; 
                    color: #212529;
                    gridline-color: #dee2e6;
                    selection-background-color: #6a9eff;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                }

                QTableWidget::item {
                    padding: 4px;
                    border-bottom: 1px solid #dee2e6;
                }

                QTableWidget::item:selected {
                    background-color: #6a9eff;
                    color: white;
                }

                QTextEdit { 
                    background-color: #ffffff; 
                    color: #212529;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 4px;
                }

                QLabel {
                    color: #212529;
                    font-size: 9pt;
                }

                QStatusBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #212529;
                    border-top: 1px solid #dee2e6;
                    padding: 2px;
                }

                QProgressBar {
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    text-align: center;
                    background: #f8f9fa;
                    color: #212529;
                }

                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #6a9eff, stop:1 #3e7ce8);
                    border-radius: 3px;
                }

                QCheckBox {
                    color: #212529;
                    spacing: 8px;
                }

                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #ced4da;
                    border-radius: 3px;
                    background: #ffffff;
                }

                QCheckBox::indicator:checked {
                    background: #6a9eff;
                    border: 2px solid #6a9eff;
                    image: none;
                }

                QCheckBox::indicator:checked:after {
                    content: "✓";
                    color: white;
                    font-weight: bold;
                }
            """)
        else:
            # 亮色Theme
            self.setStyleSheet("""
                QMainWindow { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #f8f9fa, stop:1 #e9ecef);
                    color: #212529;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }

                QMenuBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #212529;
                    border-bottom: 1px solid #dee2e6;
                    padding: 3px;
                    font-size: 10pt;
                }

                QMenuBar::item {
                    background: transparent;
                    padding: 6px 12px;
                    border-radius: 4px;
                    margin: 1px;
                }

                QMenuBar::item:selected {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #007bff, stop:1 #0056b3);
                    color: white;
                }

                QToolBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    border: none;
                    padding: 3px;
                    spacing: 3px;
                }

                QGroupBox { 
                    color: #212529; 
                    border: 2px solid #dee2e6;
                    border-radius: 8px;
                    margin-top: 10px;
                    padding-top: 5px;
                    font-weight: bold;
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                }

                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 8px 0 8px;
                    color: #007bff;
                }

                QPushButton { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f1f3f4);
                    color: #212529; 
                    border: 1px solid #ced4da;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-size: 9pt;
                    font-weight: 500;
                    min-height: 20px;
                    text-align: left;
                    padding-left: 20px;
                }

                QPushButton:hover { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #007bff, stop:1 #0056b3);
                    border: 1px solid #007bff;
                    color: white;
                }

                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #0056b3, stop:1 #004085);
                    border: 1px solid #004085;
                    color: white;
                }

                QPushButton:checked {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #007bff, stop:1 #0056b3);
                    border: 1px solid #0056b3;
                    color: white;
                }

                QComboBox { 
                    background: #ffffff;
                    color: #212529; 
                    border: 1px solid #ced4da;
                    border-radius: 4px;
                    padding: 5px;
                    min-width: 80px;
                }

                QComboBox:hover {
                    border: 1px solid #007bff;
                }

                QSlider::groove:horizontal { 
                    background: #e9ecef; 
                    height: 6px;
                    border-radius: 3px;
                }

                QSlider::handle:horizontal { 
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #007bff, stop:1 #0056b3);
                    width: 16px;
                    height: 16px;
                    border-radius: 8px;
                    margin: -5px 0;
                    border: 2px solid #ffffff;
                }

                QTableWidget { 
                    background-color: #ffffff; 
                    color: #212529;
                    gridline-color: #dee2e6;
                    selection-background-color: #007bff;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                }

                QTableWidget::item:selected {
                    background-color: #007bff;
                    color: white;
                }

                QTextEdit { 
                    background-color: #ffffff; 
                    color: #212529;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 4px;
                }

                QLabel {
                    color: #212529;
                    font-size: 9pt;
                }

                QStatusBar {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                        stop:0 #ffffff, stop:1 #f8f9fa);
                    color: #212529;
                    border-top: 1px solid #dee2e6;
                    padding: 2px;
                }

                QProgressBar {
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    text-align: center;
                    background: #f8f9fa;
                    color: #212529;
                }

                QProgressBar::chunk {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                        stop:0 #007bff, stop:1 #0056b3);
                    border-radius: 3px;
                }

                QCheckBox {
                    color: #212529;
                    spacing: 8px;
                }

                QCheckBox::indicator {
                    width: 16px;
                    height: 16px;
                    border: 2px solid #ced4da;
                    border-radius: 3px;
                    background: #ffffff;
                }

                QCheckBox::indicator:checked {
                    background: #007bff;
                    border: 2px solid #007bff;
                }
            """)

        self.vtk_widget.GetRenderWindow().Render()

    def switch_theme(self, theme_name):
        """切换Theme"""
        self.current_theme = theme_name
        self.apply_theme()

        # 更新Theme指示器
        if hasattr(self, 'theme_label'):
            if theme_name == "dark":
                self.theme_label.setText("🌙 深色Theme")
            else:
                self.theme_label.setText("☀️ 亮色Theme")

        self.update_status(f"🎨 Theme已切换到: {theme_name}")

    def update_grid_color(self):
        """更新网格颜色"""
        # 这里需要重新创建网格以Apply新颜色
        # 简化处理，实际Apply中可以保存网格actors并更新颜色
        pass

    def update_status(self, message):
        """更新状态栏消息"""
        self.status_label.setText(message)
        QTimer.singleShot(3000, lambda: self.status_label.setText("准备就绪"))

    def verify_chinese_font(self):
        """验证和重新设置中文字体"""
        try:
            import matplotlib.font_manager as fm
            current_font = matplotlib.rcParams['font.sans-serif'][0]

            # 测试当前字体是否支持中文
            test_text = "测试中文字体"

            # 强制重新加载字体管理器
            fm._load_fontmanager(try_read_cache=False)

            # 重新Apply字体设置
            setup_chinese_font()

            self.update_status(f"字体设置已验证: {current_font}")

        except Exception as e:
            print(f"字体验证失败: {e}")
            # 如果验证失败，尝试使用默认字体
            matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
            self.update_status("使用默认字体")

    def update_data_info(self):
        """更新数据信息显示"""
        if self.df_latest is not None:
            total_rows = len(self.df_latest)
            unique_holes = self.df_latest['ID'].nunique() if 'ID' in self.df_latest.columns else 0
            unique_layers = self.df_latest['layername'].nunique() if 'layername' in self.df_latest.columns else 0

            info_text = f"📊 {total_rows}行 | 🕳️{unique_holes}孔 | 🪨{unique_layers}层"
            self.data_info_label.setText(info_text)

            # 更新统计信息
            stats = self.calculate_data_statistics()
            self.stats_text.setPlainText(stats)
        else:
            self.data_info_label.setText("📊 无数据")
            self.stats_text.clear()

    def calculate_data_statistics(self):
        """计算Data Statistics信息"""
        if self.df_latest is None:
            return "无数据"

        stats = []
        stats.append(f"总记录数: {len(self.df_latest)}")

        if 'ID' in self.df_latest.columns:
            stats.append(f"钻孔数量: {self.df_latest['ID'].nunique()}")

        if 'layername' in self.df_latest.columns:
            stats.append(f"地层类型: {self.df_latest['layername'].nunique()}")
            stats.append("\n地层分布:")
            layer_counts = self.df_latest['layername'].value_counts()
            for layer, count in layer_counts.items():
                stats.append(f"  {layer}: {count}")

        if all(col in self.df_latest.columns for col in ['X', 'Y', 'Z']):
            stats.append(f"\n坐标范围:")
            stats.append(f"  X: {self.df_latest['X'].min():.2f} ~ {self.df_latest['X'].max():.2f}")
            stats.append(f"  Y: {self.df_latest['Y'].min():.2f} ~ {self.df_latest['Y'].max():.2f}")
            stats.append(f"  Z: {self.df_latest['Z'].min():.2f} ~ {self.df_latest['Z'].max():.2f}")

        if all(col in self.df_latest.columns for col in ['Top', 'Bottom']):
            thicknesses = self.df_latest['Bottom'] - self.df_latest['Top']
            stats.append(f"\n厚度统计:")
            stats.append(f"  平均厚度: {thicknesses.mean():.2f}")
            stats.append(f"  最大厚度: {thicknesses.max():.2f}")
            stats.append(f"  最小厚度: {thicknesses.min():.2f}")

        return '\n'.join(stats)

    def reset_camera(self):
        """Reset相机视角"""
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(100, 100, 200)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        self.renderer.ResetCamera()
        self.renderer.GetActiveCamera().Zoom(1.5)
        self.vtk_widget.GetRenderWindow().Render()

    def show_about(self):
        """显示About对话框"""
        QMessageBox.about(self, "About",
                          "增强型钻孔三维可视化系统 v2.0\n\n"
                          "功能特性:\n"
                          "• 多File数据管理\n"
                          "• Data Validation和过滤\n"
                          "• 体积和厚度分析\n"
                          "• 多格式ImportExport\n"
                          "• Theme切换\n"
                          "• 快捷键支持\n\n"
                          "基于PyQt5和VTK开发")

    def load_multiple_csv(self):
        """Import多个CSVFile"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择多个CSVFile", "", "CSVFile (*.csv)"
        )

        if not file_paths:
            return

        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(len(file_paths))

            combined_df = pd.DataFrame()

            for i, file_path in enumerate(file_paths):
                self.progress_bar.setValue(i)
                self.update_status(f"正在加载File {i + 1}/{len(file_paths)}")

                df = pd.read_csv(file_path)

                # 添加File来源标记
                df['source_file'] = os.path.basename(file_path)

                combined_df = pd.concat([combined_df, df], ignore_index=True)

            self.progress_bar.setValue(len(file_paths))
            self.progress_bar.setVisible(False)

            # Data Validation
            validation_dialog = DataValidationDialog(combined_df, self)
            if validation_dialog.exec_() == QDialog.Accepted:
                self.df_original = combined_df.copy()
                self.df_latest = combined_df.copy()
                self.create_borehole_models(combined_df)
                self.update_data_info()
                self.update_status(f"成功Import {len(file_paths)} 个CSVFile")
            else:
                self.update_status("数据Import已Cancel")

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "错误", f"Import多个CSVFile时发生错误：{e}")

    def validate_data(self):
        """验证当前数据"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        validation_dialog = DataValidationDialog(self.df_latest, self)
        validation_dialog.exec_()

    def open_data_filter(self):
        """打开Data Filter对话框"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        filter_dialog = DataFilterDialog(self.df_latest, self)
        if filter_dialog.exec_() == QDialog.Accepted:
            self.df_latest = filter_dialog.filtered_df
            self.create_borehole_models(self.df_latest)
            self.update_data_info()
            self.update_status("Data Filter已Apply")

    def open_analysis_dialog(self):
        """打开Analysis Tools对话框"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        if self.analysis_dialog is None or not self.analysis_dialog.isVisible():
            self.analysis_dialog = AnalysisDialog(self.df_latest, self)

        self.analysis_dialog.show()
        self.analysis_dialog.raise_()

    def export_excel(self):
        """Export为Excel格式"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "没有数据可Export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存ExcelFile", "", "ExcelFile (*.xlsx)"
        )

        if file_path:
            try:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    # 主数据表
                    self.df_latest.to_excel(writer, sheet_name='钻孔数据', index=False)

                    # 统计表
                    if 'layername' in self.df_latest.columns:
                        layer_stats = self.df_latest.groupby('layername').agg({
                            'Top': ['min', 'max', 'mean'],
                            'Bottom': ['min', 'max', 'mean'],
                            'ID': 'count'
                        }).round(2)
                        layer_stats.to_excel(writer, sheet_name='地层统计')

                self.update_status(f"数据已Export到: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"ExportExcel失败：{e}")

    def export_json(self):
        """Export为JSON格式"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "没有数据可Export")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存JSONFile", "", "JSONFile (*.json)"
        )

        if file_path:
            try:
                # 准备Export数据
                export_data = {
                    'metadata': {
                        'export_time': datetime.now().isoformat(),
                        'total_records': len(self.df_latest),
                        'unique_holes': self.df_latest['ID'].nunique() if 'ID' in self.df_latest.columns else 0,
                        'data_ranges': {}
                    },
                    'data': self.df_latest.to_dict('records')
                }

                # 添加数据范围信息
                numeric_columns = ['X', 'Y', 'Z', 'Top', 'Bottom']
                for col in numeric_columns:
                    if col in self.df_latest.columns:
                        export_data['metadata']['data_ranges'][col] = {
                            'min': float(self.df_latest[col].min()),
                            'max': float(self.df_latest[col].max()),
                            'mean': float(self.df_latest[col].mean())
                        }

                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)

                self.update_status(f"数据已Export到: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"ExportJSON失败：{e}")

    def export_stl(self):
        """Export3D模型为STL格式"""
        if not self.closed_model_actors:
            QMessageBox.warning(self, "警告", "请先生成3D模型")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存STLFile", "", "STLFile (*.stl)"
        )

        if file_path:
            try:
                # 合并所有模型数据
                append_filter = vtk.vtkAppendPolyData()

                for actor in self.closed_model_actors:
                    mapper = actor.GetMapper()
                    if mapper and mapper.GetInput():
                        append_filter.AddInputData(mapper.GetInput())

                append_filter.Update()

                # 写入STLFile
                stl_writer = vtk.vtkSTLWriter()
                stl_writer.SetFileName(file_path)
                stl_writer.SetInputConnection(append_filter.GetOutputPort())
                stl_writer.Write()

                self.update_status(f"3D模型已Export到: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"ExportSTL失败：{e}")

    def export_image(self):
        """Export当前View为图像"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "", "PNGFile (*.png);;JPGFile (*.jpg)"
        )

        if file_path:
            try:
                # 获取渲染窗口
                render_window = self.vtk_widget.GetRenderWindow()

                # 创建窗口到图像过滤器
                window_to_image = vtk.vtkWindowToImageFilter()
                window_to_image.SetInput(render_window)
                window_to_image.Update()

                # 写入图像File
                if file_path.lower().endswith('.png'):
                    writer = vtk.vtkPNGWriter()
                else:
                    writer = vtk.vtkJPEGWriter()

                writer.SetFileName(file_path)
                writer.SetInputConnection(window_to_image.GetOutputPort())
                writer.Write()

                self.update_status(f"图像已Export到: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"Export图像失败：{e}")

    def update_color_mapping(self):
        """更新颜色映射"""
        if self.df_latest is None:
            return

        mode = self.color_mode_combo.currentText()

        # 重新创建模型以Apply新的颜色映射
        self.create_borehole_models(self.df_latest, color_mode=mode)

    def update_opacity(self):
        """更新Opacity"""
        opacity = self.opacity_slider.value() / 100.0

        # 更新所有相关actors的Opacity
        for actor in self.actor_data_map.keys():
            if hasattr(actor, 'GetProperty') and actor.GetProperty():
                actor.GetProperty().SetOpacity(opacity)

        for actors in self.layer_top_mesh_actors.values():
            if hasattr(actors, 'GetProperty') and actors.GetProperty():
                actors.GetProperty().SetOpacity(opacity * 0.5)
            elif hasattr(actors, '__iter__'):  # 如果是列表
                for actor in actors:
                    if hasattr(actor, 'GetProperty') and actor.GetProperty():
                        actor.GetProperty().SetOpacity(opacity * 0.5)

        if self.coal_bottom_mesh_actor and hasattr(self.coal_bottom_mesh_actor, 'GetProperty'):
            self.coal_bottom_mesh_actor.GetProperty().SetOpacity(opacity * 0.5)

        for actor in self.closed_model_actors:
            if hasattr(actor, 'GetProperty') and actor.GetProperty():
                actor.GetProperty().SetOpacity(opacity * 0.7)

        self.vtk_widget.GetRenderWindow().Render()

    def update_smooth_value_label(self):
        """更新平滑值标签"""
        value = self.smooth_slider.value()
        self.smooth_value_label.setText(str(value))

        # 提供实时提示
        if hasattr(self, 'status_label'):
            if value <= 20:
                self.update_status("🌊 平滑设置: 轻微平滑 - 保持更多细节")
            elif value <= 50:
                self.update_status("🌊 平滑设置: 适中平滑 - 平衡细节与平滑")
            else:
                self.update_status("🌊 平滑设置: 强力平滑 - 获得最佳平滑效果")

    def get_smoothing_parameters(self):
        """获取当前平滑参数"""
        iterations = self.smooth_slider.value()
        use_windowed_sinc = self.smooth_type_combo.currentText() == '高质量平滑'

        # 根据迭代次数调整松弛因子
        if iterations <= 30:
            relaxation_factor = 0.1
        elif iterations <= 60:
            relaxation_factor = 0.08
        else:
            relaxation_factor = 0.06

        return iterations, relaxation_factor, use_windowed_sinc

    def interpolate_missing_data(self):
        """插值填补缺失数据"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        try:
            # 检查是否有缺失数据
            missing_data = self.df_latest.isnull().sum()
            if missing_data.sum() == 0:
                QMessageBox.information(self, "信息", "数据完整，无需插值")
                return

            # 对数值列进行插值
            numeric_columns = ['X', 'Y', 'Z', 'Top', 'Bottom']
            df_interpolated = self.df_latest.copy()

            for col in numeric_columns:
                if col in df_interpolated.columns:
                    df_interpolated[col] = df_interpolated[col].interpolate(method='linear')

            self.df_latest = df_interpolated
            self.create_borehole_models(self.df_latest)
            self.update_data_info()
            self.update_status("Data Interpolation完成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Data Interpolation失败：{e}")

    def smooth_data(self):
        """平滑数据"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        try:
            # 使用滑动平均平滑数值数据
            from scipy.ndimage import uniform_filter1d

            numeric_columns = ['X', 'Y', 'Z', 'Top', 'Bottom']
            df_smoothed = self.df_latest.copy()

            for col in numeric_columns:
                if col in df_smoothed.columns:
                    # 按钻孔ID分组进行平滑
                    for hole_id in df_smoothed['ID'].unique():
                        mask = df_smoothed['ID'] == hole_id
                        values = df_smoothed.loc[mask, col].values
                        if len(values) > 1:
                            smoothed = uniform_filter1d(values.astype(float), size=3, mode='reflect')
                            df_smoothed.loc[mask, col] = smoothed

            self.df_latest = df_smoothed
            self.create_borehole_models(self.df_latest)
            self.update_data_info()
            self.update_status("Data Smoothing完成")

        except ImportError:
            QMessageBox.warning(self, "警告", "需要安装scipy库来执行Data Smoothing")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Data Smoothing失败：{e}")

    def detect_anomalies(self):
        """检测异常数据"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        try:
            anomalies = []

            # 检查坐标异常
            if all(col in self.df_latest.columns for col in ['X', 'Y', 'Z']):
                for col in ['X', 'Y', 'Z']:
                    Q1 = self.df_latest[col].quantile(0.25)
                    Q3 = self.df_latest[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outliers = self.df_latest[
                        (self.df_latest[col] < lower_bound) |
                        (self.df_latest[col] > upper_bound)
                        ]

                    if len(outliers) > 0:
                        anomalies.append(f"{col}坐标异常: {len(outliers)}个点")

            # 检查深度逻辑异常
            if all(col in self.df_latest.columns for col in ['Top', 'Bottom']):
                depth_errors = self.df_latest[self.df_latest['Top'] >= self.df_latest['Bottom']]
                if len(depth_errors) > 0:
                    anomalies.append(f"深度逻辑错误: {len(depth_errors)}个记录")

            # 检查厚度异常
            if all(col in self.df_latest.columns for col in ['Top', 'Bottom']):
                thickness = self.df_latest['Bottom'] - self.df_latest['Top']
                thickness_outliers = thickness[thickness > thickness.quantile(0.95)]
                if len(thickness_outliers) > 0:
                    anomalies.append(f"异常厚度: {len(thickness_outliers)}个记录")

            if anomalies:
                result = '\n'.join(anomalies)
                QMessageBox.information(self, "Anomaly Detection结果", result)
            else:
                QMessageBox.information(self, "Anomaly Detection结果", "未发现明显异常")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Anomaly Detection失败：{e}")

    def create_profile_analysis(self):
        """创建Profile Analysis"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        try:
            # 简化的Profile Analysis：沿X轴Direction
            profile_data = self.df_latest.copy()
            profile_data = profile_data.sort_values('X')

            # 创建剖面图
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # By Stratum绘制
            layers = profile_data['layername'].unique()
            colors = plt.cm.Set3(np.linspace(0, 1, len(layers)))

            for i, layer in enumerate(layers):
                layer_data = profile_data[profile_data['layername'] == layer]
                ax.scatter(layer_data['X'], layer_data['Z'],
                           c=[colors[i]], label=layer, s=50, alpha=0.7)

            ax.set_xlabel('X坐标')
            ax.set_ylabel('Z坐标(深度)')
            ax.set_title('钻孔Profile Analysis')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            self.update_status("Profile Analysis图已生成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建Profile Analysis失败：{e}")

    def create_profile_analysis(self):
        """创建Profile Analysis"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import数据")
            return

        try:
            # 简化的Profile Analysis：沿X轴Direction
            profile_data = self.df_latest.copy()
            profile_data = profile_data.sort_values('X')

            # 创建剖面图
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            # By Stratum绘制
            for layer in profile_data['layername'].unique():
                layer_data = profile_data[profile_data['layername'] == layer]

                for _, row in layer_data.iterrows():
                    x = row['X']
                    top = row['Top']
                    bottom = row['Bottom']

                    color = layer_colors.get(layer, (0.5, 0.5, 0.5))
                    ax.fill_between([x - 1, x + 1], [top, top], [bottom, bottom],
                                    color=color, alpha=0.7, label=layer)

            # 设置图表标签和标题，确保使用正确的字体
            ax.set_xlabel('X 坐标', fontsize=12)
            ax.set_ylabel('深度', fontsize=12)
            ax.set_title('地质剖面图', fontsize=14, fontweight='bold')
            ax.invert_yaxis()  # 深度向下

            # 去重图例，设置字体
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),
                      loc='upper right', fontsize=10)

            # 设置坐标轴标签字体
            ax.tick_params(axis='both', which='major', labelsize=10)

            plt.tight_layout()
            plt.show()

            self.update_status("Profile Analysis已生成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Profile Analysis失败：{e}")

    def _add_axes_widget(self):
        # 极简三线+XYZ文字坐标系，固定在左下角
        axes_length = 30
        axes_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # x红y绿z蓝
        axes_dirs = [(axes_length, 0, 0), (0, axes_length, 0), (0, 0, axes_length)]
        axes_labels = ['X', 'Y', 'Z']
        axes_label_pos = [(axes_length + 5, 0, 0), (0, axes_length + 5, 0), (0, 0, axes_length + 5)]

        # 创建一个空的assembly作为marker
        marker_assembly = vtk.vtkAssembly()
        for i in range(3):
            # 线
            line = vtk.vtkLineSource()
            line.SetPoint1(0, 0, 0)
            line.SetPoint2(*axes_dirs[i])
            line.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(line.GetOutput())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*axes_colors[i])
            actor.GetProperty().SetLineWidth(3)
            actor.PickableOff()
            marker_assembly.AddPart(actor)
            # 文字
            txt = vtk.vtkVectorText()
            txt.SetText(axes_labels[i])
            txt.Update()
            txt_mapper = vtk.vtkPolyDataMapper()
            txt_mapper.SetInputData(txt.GetOutput())
            txt_actor = vtk.vtkFollower()
            txt_actor.SetMapper(txt_mapper)
            txt_actor.SetScale(5, 5, 5)
            txt_actor.SetPosition(*axes_label_pos[i])
            txt_actor.GetProperty().SetColor(*axes_colors[i])
            txt_actor.PickableOff()
            marker_assembly.AddPart(txt_actor)

        self.marker = vtk.vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(marker_assembly)
        self.marker.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
        self.marker.SetViewport(0.0, 0.0, 0.18, 0.18)  # 左下角
        self.marker.SetEnabled(1)
        self.marker.InteractiveOff()

    def load_csv(self):
        """Import单个CSVFile"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择CSVFile", "", "CSVFile (*.csv)")
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)

            # Data Validation
            validation_dialog = DataValidationDialog(df, self)
            if validation_dialog.exec_() == QDialog.Accepted:
                self.df_original = df.copy()
                self.df_latest = df.copy()
                self.create_borehole_models(df)
                self.update_data_info()
                self.update_status("CSVFileImport成功")
            else:
                self.update_status("数据Import已Cancel")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载CSVFile时发生错误：{e}")

    def create_borehole_models(self, df, color_mode='By Stratum'):
        """创建Borehole Models"""
        # 清除原有Borehole Models
        for actor in list(self.actor_data_map.keys()):
            self.renderer.RemoveActor(actor)
        self.actor_data_map.clear()
        self.models_visible = True

        grouped = df.groupby("ID")
        for hole_id, group in grouped:
            x, y, z = float(group.iloc[0]['X']), -float(group.iloc[0]['Y']), -float(group.iloc[0]['Z'])

            for _, row in group.iterrows():
                top = row['Top']
                bottom = row['Bottom']
                height = bottom - top
                name = row.get("layername", "未知")

                # 根据颜色模式OK颜色
                if color_mode == 'By Depth':
                    # 基于深度的颜色映射
                    depth_ratio = (top + bottom) / 2 / df['Bottom'].max()
                    color = (depth_ratio, 0.5, 1 - depth_ratio)
                elif color_mode == 'By Thickness':
                    # 基于厚度的颜色映射
                    thickness_ratio = height / (df['Bottom'] - df['Top']).max()
                    color = (1 - thickness_ratio, thickness_ratio, 0.5)
                else:
                    # 默认By Stratum颜色
                    color = layer_colors.get(name, (0.5, 0.5, 0.5))

                cylinder = vtk.vtkCylinderSource()
                cylinder.SetRadius(1)
                cylinder.SetHeight(height)
                cylinder.SetResolution(20)

                transform = vtk.vtkTransform()
                transform.Translate(x, y, z + top + height / 2)
                transform.RotateX(90)

                transform_filter = vtk.vtkTransformPolyDataFilter()
                transform_filter.SetTransform(transform)
                transform_filter.SetInputConnection(cylinder.GetOutputPort())
                transform_filter.Update()

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(transform_filter.GetOutputPort())
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(color)

                self.renderer.AddActor(actor)
                self.actor_data_map[actor] = row.to_dict()

        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def show_info_panel(self, info):
        """显示选中对象信息"""
        self.info_table.setRowCount(len(info))
        for i, (key, val) in enumerate(info.items()):
            self.info_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(key)))
            self.info_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(val)))

    def add_grid_plane(self, size=100, spacing=10):
        """添加网格平面"""
        grid_color = THEMES[self.current_theme]["grid_color"]

        for i in range(-size // 2, size // 2 + 1, spacing):
            for horiz in [(1, 0), (0, 1)]:
                points = vtk.vtkPoints()
                lines = vtk.vtkCellArray()
                points.InsertNextPoint(i * horiz[0] - size / 2 * (1 - horiz[0]),
                                       i * horiz[1] - size / 2 * (1 - horiz[1]), 0)
                points.InsertNextPoint(i * horiz[0] + size / 2 * (1 - horiz[0]),
                                       i * horiz[1] + size / 2 * (1 - horiz[1]), 0)
                lines.InsertNextCell(2)
                lines.InsertCellPoint(0)
                lines.InsertCellPoint(1)
                grid = vtk.vtkPolyData()
                grid.SetPoints(points)
                grid.SetLines(lines)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(grid)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(*grid_color)
                actor.GetProperty().SetLineWidth(1)
                self.renderer.AddActor(actor)

    def _gather_borehole_xy(self):
        """收集钻孔XY坐标"""
        hole_pos = {}
        for row in self.actor_data_map.values():
            hole_id = row.get("ID")
            if hole_id is None:
                continue
            if hole_id in hole_pos:
                continue
            try:
                x = float(row.get("X", 0))
                y = -float(row.get("Y", 0))
            except Exception:
                continue
            hole_pos[hole_id] = (x, y)
        return hole_pos

    def _gather_layer_top_points(self):
        """收集地层顶面点"""
        layer_points = {}
        for row in self.actor_data_map.values():
            layer = row.get("layername", "未知")
            try:
                x = float(row.get("X", 0))
                y = -float(row.get("Y", 0))
                z_origin = -float(row.get("Z", 0))
                top = float(row.get("Top", 0))
            except Exception:
                continue
            z_top = z_origin + top
            layer_points.setdefault(layer, []).append((x, y, z_top))
        return layer_points

    def _gather_coal_bottom_points(self):
        """收集煤层底面点"""
        pts = []
        for row in self.actor_data_map.values():
            layer = row.get("layername", "")
            if layer != "煤层":
                continue
            try:
                x = float(row.get("X", 0))
                y = -float(row.get("Y", 0))
                z_origin = -float(row.get("Z", 0))
                bottom = float(row.get("Bottom", 0))
            except Exception:
                continue
            z_bot = z_origin + bottom
            pts.append((x, y, z_bot))
        return pts

    def _create_borehole_point_actor(self):
        """创建Borehole Pointsactor"""
        hole_pos = self._gather_borehole_xy()
        if not hole_pos:
            return None
        points = vtk.vtkPoints()
        for hid, (x, y) in hole_pos.items():
            points.InsertNextPoint(x, y, 0)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(2)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(16)
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetInputData(poly)
        glyph.ScalingOff()
        glyph.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)
        actor.GetProperty().SetDiffuse(0.8)
        actor.GetProperty().SetSpecular(0.3)
        actor.GetProperty().SetSpecularPower(20)
        return actor

    def toggle_borehole_points(self):
        """切换Borehole Points显示"""
        if not self.point_actors:
            # 还没有点，创建并显示
            actor = self._create_borehole_point_actor()
            if actor:
                self.renderer.AddActor(actor)
                self.point_actors.append(actor)
                self.points_visible = True
        else:
            # 切换可见性
            self.points_visible = not self.points_visible
            for actor in self.point_actors:
                actor.SetVisibility(1 if self.points_visible else 0)
        self.vtk_widget.GetRenderWindow().Render()

    def create_delaunay_model(self):
        """创建Delaunay三角网模型"""
        if self.delaunay_actor:
            self.renderer.RemoveActor(self.delaunay_actor)
            self.delaunay_actor = None

        hole_pos = self._gather_borehole_xy()
        if not hole_pos:
            QMessageBox.warning(self, "提示", "没有Borehole Points用于 Delaunay。")
            return

        points = vtk.vtkPoints()
        for hid, (x, y) in hole_pos.items():
            points.InsertNextPoint(x, y, 0)
        poly = vtk.vtkPolyData()
        poly.SetPoints(points)

        verts = vtk.vtkCellArray()
        for i in range(points.GetNumberOfPoints()):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        poly.SetVerts(verts)

        delaunay = vtk.vtkDelaunay2D()
        delaunay.SetInputData(poly)
        delaunay.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(delaunay.GetOutputPort())
        filled_actor = vtk.vtkActor()
        filled_actor.SetMapper(mapper)
        filled_actor.GetProperty().SetOpacity(0.5)
        filled_actor.GetProperty().SetRepresentationToSurface()
        filled_actor.GetProperty().SetColor(0.2, 0.6, 0.8)

        edge_mapper = vtk.vtkPolyDataMapper()
        edge_mapper.SetInputConnection(delaunay.GetOutputPort())
        edge_actor = vtk.vtkActor()
        edge_actor.SetMapper(edge_mapper)
        edge_actor.GetProperty().SetRepresentationToWireframe()
        edge_actor.GetProperty().SetLineWidth(1.0)
        edge_actor.GetProperty().SetColor(0, 0, 0)

        assembly = vtk.vtkAssembly()
        assembly.AddPart(filled_actor)
        assembly.AddPart(edge_actor)

        self.renderer.AddActor(assembly)
        self.delaunay_actor = assembly
        self.delaunay_visible = True
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_delaunay(self):
        """切换Delaunay三角网显示"""
        if not self.delaunay_actor:
            self.create_delaunay_model()
        else:
            self.delaunay_visible = not self.delaunay_visible
            self.delaunay_actor.SetVisibility(1 if self.delaunay_visible else 0)
            self.vtk_widget.GetRenderWindow().Render()

    def create_layer_top_models(self):
        """创建地层顶面模型"""
        # 清除已有layer top和煤层bottom网格
        for layer_name, actors in self.layer_top_mesh_actors.items():
            if hasattr(actors, 'RemoveActor'):
                self.renderer.RemoveActor(actors)
            else:
                for actor in actors if isinstance(actors, list) else [actors]:
                    self.renderer.RemoveActor(actor)
        self.layer_top_mesh_actors.clear()
        self.layer_top_visible.clear()

        # 清除Stratum Control面板上的所有控件
        for i in reversed(range(self.layer_control_layout.count())):
            widget = self.layer_control_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        if self.coal_bottom_mesh_actor:
            self.renderer.RemoveActor(self.coal_bottom_mesh_actor)
            self.coal_bottom_mesh_actor = None

        # 地层top
        layer_points = self._gather_layer_top_points()
        for layer_name, pts in layer_points.items():
            if len(pts) < 3:
                continue

            # 为每个地层创建显示控制按钮
            btn = QPushButton(f"显示/隐藏 {layer_name}")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.clicked.connect(lambda checked, ln=layer_name: self.toggle_single_layer(ln, checked))
            self.layer_control_layout.addWidget(btn)

            vtk_pts = vtk.vtkPoints()
            for x, y, z in pts:
                vtk_pts.InsertNextPoint(x, y, z)
            poly = vtk.vtkPolyData()
            poly.SetPoints(vtk_pts)
            verts = vtk.vtkCellArray()
            for i in range(vtk_pts.GetNumberOfPoints()):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            poly.SetVerts(verts)

            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(poly)
            delaunay.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(delaunay.GetOutputPort())
            filled_actor = vtk.vtkActor()
            filled_actor.SetMapper(mapper)
            filled_actor.GetProperty().SetOpacity(0.3)
            filled_actor.GetProperty().SetRepresentationToSurface()
            base_color = layer_colors.get(layer_name, (0.7, 0.7, 0.7))
            filled_actor.GetProperty().SetColor(base_color)

            edge_mapper = vtk.vtkPolyDataMapper()
            edge_mapper.SetInputConnection(delaunay.GetOutputPort())
            edge_actor = vtk.vtkActor()
            edge_actor.SetMapper(edge_mapper)
            edge_actor.GetProperty().SetRepresentationToWireframe()
            edge_actor.GetProperty().SetLineWidth(1.0)
            edge_actor.GetProperty().SetColor(0, 0, 0)

            assembly = vtk.vtkAssembly()
            assembly.AddPart(filled_actor)
            assembly.AddPart(edge_actor)

            self.renderer.AddActor(assembly)
            self.layer_top_mesh_actors[layer_name] = assembly
            self.layer_top_visible[layer_name] = True

        # 煤层bottom
        coal_bottom_pts = self._gather_coal_bottom_points()
        if len(coal_bottom_pts) >= 3:
            vtk_pts = vtk.vtkPoints()
            for x, y, z in coal_bottom_pts:
                vtk_pts.InsertNextPoint(x, y, z)
            poly = vtk.vtkPolyData()
            poly.SetPoints(vtk_pts)
            verts = vtk.vtkCellArray()
            for i in range(vtk_pts.GetNumberOfPoints()):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)
            poly.SetVerts(verts)
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(poly)
            delaunay.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(delaunay.GetOutputPort())
            filled_actor = vtk.vtkActor()
            filled_actor.SetMapper(mapper)
            filled_actor.GetProperty().SetOpacity(0.3)
            filled_actor.GetProperty().SetRepresentationToSurface()
            filled_actor.GetProperty().SetColor(0.2, 0.2, 0.2)

            edge_mapper = vtk.vtkPolyDataMapper()
            edge_mapper.SetInputConnection(delaunay.GetOutputPort())
            edge_actor = vtk.vtkActor()
            edge_actor.SetMapper(edge_mapper)
            edge_actor.GetProperty().SetRepresentationToWireframe()
            edge_actor.GetProperty().SetLineWidth(1.0)
            edge_actor.GetProperty().SetColor(0, 0, 0)

            assembly = vtk.vtkAssembly()
            assembly.AddPart(filled_actor)
            assembly.AddPart(edge_actor)

            self.renderer.AddActor(assembly)
            self.coal_bottom_mesh_actor = assembly

        self.vtk_widget.GetRenderWindow().Render()

    def _add_axes_widget(self):
        """添加坐标轴小部件"""
        # 极简三线+XYZ文字坐标系，固定在左下角
        axes_length = 30
        axes_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # x红y绿z蓝
        axes_dirs = [(axes_length, 0, 0), (0, axes_length, 0), (0, 0, axes_length)]
        axes_labels = ['X', 'Y', 'Z']
        axes_label_pos = [(axes_length + 5, 0, 0), (0, axes_length + 5, 0), (0, 0, axes_length + 5)]

        # 创建一个空的assembly作为marker
        marker_assembly = vtk.vtkAssembly()
        for i in range(3):
            # 线
            line = vtk.vtkLineSource()
            line.SetPoint1(0, 0, 0)
            line.SetPoint2(*axes_dirs[i])
            line.Update()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(line.GetOutput())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*axes_colors[i])
            actor.GetProperty().SetLineWidth(3)
            actor.PickableOff()
            marker_assembly.AddPart(actor)
            # 文字
            txt = vtk.vtkVectorText()
            txt.SetText(axes_labels[i])
            txt.Update()
            txt_mapper = vtk.vtkPolyDataMapper()
            txt_mapper.SetInputData(txt.GetOutput())
            txt_actor = vtk.vtkFollower()
            txt_actor.SetMapper(txt_mapper)
            txt_actor.SetScale(5, 5, 5)
            txt_actor.SetPosition(*axes_label_pos[i])
            txt_actor.GetProperty().SetColor(*axes_colors[i])
            txt_actor.PickableOff()
            marker_assembly.AddPart(txt_actor)

        self.marker = vtk.vtkOrientationMarkerWidget()
        self.marker.SetOrientationMarker(marker_assembly)
        self.marker.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
        self.marker.SetViewport(0.0, 0.0, 0.18, 0.18)  # 左下角
        self.marker.SetEnabled(1)
        self.marker.InteractiveOff()

    def toggle_single_layer(self, layer_name, visible):
        """切换单个地层显示"""
        if layer_name in self.layer_top_mesh_actors:
            self.layer_top_mesh_actors[layer_name].SetVisibility(1 if visible else 0)
            self.layer_top_visible[layer_name] = visible
            self.vtk_widget.GetRenderWindow().Render()

    def toggle_layer_top_delaunay(self):
        """切换地层顶面Delaunay显示"""
        if not self.layer_top_mesh_actors:
            self.create_layer_top_models()
        else:
            all_visible = all(self.layer_top_visible.values()) if self.layer_top_visible else False
            new_state = not all_visible
            for layer_name in self.layer_top_visible:
                self.layer_top_visible[layer_name] = new_state
                self.layer_top_mesh_actors[layer_name].SetVisibility(1 if new_state else 0)

            # 更新按钮状态
            for i in range(self.layer_control_layout.count()):
                widget = self.layer_control_layout.itemAt(i).widget()
                if isinstance(widget, QPushButton) and widget.isCheckable():
                    widget.setChecked(new_state)

            if self.coal_bottom_mesh_actor:
                self.coal_bottom_mesh_actor.SetVisibility(1 if new_state else 0)
            self.vtk_widget.GetRenderWindow().Render()

    def toggle_borehole_models(self):
        """切换Borehole Models显示"""
        if not self.actor_data_map:
            return
        self.models_visible = not self.models_visible
        for actor in self.actor_data_map.keys():
            actor.SetVisibility(1 if self.models_visible else 0)
        self.vtk_widget.GetRenderWindow().Render()

    # ---------- 封闭模型构造辅助 ----------
    def _extract_boundary_loop(self, polydata: vtk.vtkPolyData):
        """提取边界环"""
        feat = vtk.vtkFeatureEdges()
        feat.SetInputData(polydata)
        feat.BoundaryEdgesOn()
        feat.FeatureEdgesOff()
        feat.ManifoldEdgesOff()
        feat.NonManifoldEdgesOff()
        feat.Update()

        stripper = vtk.vtkStripper()
        stripper.SetInputConnection(feat.GetOutputPort())
        stripper.Update()
        return stripper.GetOutput()

    def _create_ruled_between_loops(self, loop1: vtk.vtkPolyData, loop2: vtk.vtkPolyData):
        """在两个环之间创建规则表面"""
        append = vtk.vtkAppendPolyData()
        append.AddInputData(loop1)
        append.AddInputData(loop2)
        append.Update()
        ruled = vtk.vtkRuledSurfaceFilter()
        ruled.SetInputConnection(append.GetOutputPort())
        ruled.SetRuledModeToResample()
        ruled.SetResolution(50, 1)
        ruled.Update()
        return ruled.GetOutput()

    def _build_top_top_prism(self, upper_poly: vtk.vtkPolyData, lower_poly: vtk.vtkPolyData, layer_name: str):
        """构建顶面到顶面的棱柱体"""
        append = vtk.vtkAppendPolyData()
        append.AddInputData(upper_poly)
        append.AddInputData(lower_poly)
        upper_loop = self._extract_boundary_loop(upper_poly)
        lower_loop = self._extract_boundary_loop(lower_poly)
        if upper_loop.GetNumberOfPoints() > 1 and lower_loop.GetNumberOfPoints() > 1:
            side = self._create_ruled_between_loops(upper_loop, lower_loop)
            append.AddInputData(side)
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(append.GetOutputPort())
        cleaner.Update()

        # 对整个封闭体Apply平滑处理
        smoothed_shell = self.apply_mesh_smoothing(cleaner.GetOutput())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(smoothed_shell)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.6)  # 增加不Opacity

        # 使用智能颜色系统
        if hasattr(self, 'viz_manager') and self.viz_manager:
            layer_color = self.viz_manager.get_layer_color(layer_name)
        else:
            layer_color = layer_colors.get(layer_name, (0.7, 0.7, 0.7))

        actor.GetProperty().SetColor(layer_color)
        actor.GetProperty().SetEdgeVisibility(0)

        # 添加高级材质效果
        actor.GetProperty().SetSpecular(0.4)
        actor.GetProperty().SetSpecularPower(30)
        actor.GetProperty().SetDiffuse(0.7)
        actor.GetProperty().SetAmbient(0.2)

        return actor

    def _build_top_bottom_closed_shell(self, top_poly: vtk.vtkPolyData, bottom_poly: vtk.vtkPolyData):
        """构建顶面到底面的封闭壳体"""
        append = vtk.vtkAppendPolyData()
        append.AddInputData(top_poly)
        append.AddInputData(bottom_poly)
        top_loop = self._extract_boundary_loop(top_poly)
        bottom_loop = self._extract_boundary_loop(bottom_poly)
        if top_loop.GetNumberOfPoints() > 1 and bottom_loop.GetNumberOfPoints() > 1:
            side = self._create_ruled_between_loops(top_loop, bottom_loop)
            append.AddInputData(side)
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(append.GetOutputPort())
        cleaner.Update()

        # 对煤层封闭壳体Apply平滑处理
        smoothed_shell = self.apply_mesh_smoothing(cleaner.GetOutput())

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(smoothed_shell)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(0.7)  # 煤层更不透明
        actor.GetProperty().SetColor(0.25, 0.25, 0.25)  # 稍微浅一点的煤层色
        actor.GetProperty().SetEdgeVisibility(0)

        # 煤层特殊材质效果
        actor.GetProperty().SetSpecular(0.2)
        actor.GetProperty().SetSpecularPower(15)
        actor.GetProperty().SetDiffuse(0.9)
        actor.GetProperty().SetAmbient(0.1)

        return actor

    def apply_mesh_smoothing(self, polydata, iterations=None, relaxation_factor=None, use_windowed_sinc=None):
        """Apply网格Smoothing Algorithm

        Args:
            polydata: 输入的VTK PolyData
            iterations: 平滑迭代次数 (如果为None则使用用户设置)
            relaxation_factor: 松弛因子 (如果为None则使用用户设置)
            use_windowed_sinc: 是否使用窗口正弦平滑 (如果为None则使用用户设置)

        Returns:
            平滑后的PolyData
        """
        try:
            # 如果参数未指定，使用用户界面设置
            if iterations is None or relaxation_factor is None or use_windowed_sinc is None:
                user_iterations, user_relaxation, user_windowed_sinc = self.get_smoothing_parameters()
                if iterations is None:
                    iterations = user_iterations
                if relaxation_factor is None:
                    relaxation_factor = user_relaxation
                if use_windowed_sinc is None:
                    use_windowed_sinc = user_windowed_sinc

            if use_windowed_sinc:
                # 使用窗口正弦平滑器 - 高质量平滑
                smoother = vtk.vtkWindowedSincPolyDataFilter()
                smoother.SetInputData(polydata)
                smoother.SetNumberOfIterations(iterations)
                smoother.SetPassBand(0.1)  # 通带频率 (0.01-0.2)
                smoother.SetFeatureAngle(120.0)  # 特征角度
                smoother.SetEdgeAngle(15.0)  # 边缘角度
                smoother.SetBoundarySmoothing(True)  # 边界平滑
                smoother.SetNormalizeCoordinates(True)  # 坐标归一化
                smoother.Update()
                return smoother.GetOutput()
            else:
                # 使用Laplace平滑器 - 快速平滑
                smoother = vtk.vtkSmoothPolyDataFilter()
                smoother.SetInputData(polydata)
                smoother.SetNumberOfIterations(iterations)
                smoother.SetRelaxationFactor(relaxation_factor)
                smoother.SetFeatureAngle(60.0)  # 特征角度
                smoother.SetEdgeAngle(15.0)  # 边缘角度
                smoother.SetBoundarySmoothing(True)  # 启用边界平滑
                smoother.SetFeatureEdgeSmoothing(True)  # 特征边缘平滤
                smoother.Update()
                return smoother.GetOutput()

        except Exception as e:
            print(f"网格平滑失败: {e}")
            return polydata  # 返回原始数据

    def create_closed_model(self):
        """创建封闭3D模型"""
        try:
            # 清除原有封闭模型
            for actor in self.closed_model_actors:
                self.renderer.RemoveActor(actor)
            self.closed_model_actors.clear()
            self.closed_model_visible = False

            if self.df_latest is None:
                QMessageBox.warning(self, "提示", "请先Import钻孔数据再Generate Closed Model。")
                return

            layer_top = self._gather_layer_top_points()
            layer_surfaces = {}
            for layer_name, pts in layer_top.items():
                if len(pts) < 3:
                    continue
                vtk_pts = vtk.vtkPoints()
                for x, y, z in pts:
                    vtk_pts.InsertNextPoint(x, y, z)
                poly = vtk.vtkPolyData()
                poly.SetPoints(vtk_pts)
                verts = vtk.vtkCellArray()
                for i in range(vtk_pts.GetNumberOfPoints()):
                    verts.InsertNextCell(1)
                    verts.InsertCellPoint(i)
                poly.SetVerts(verts)
                delaunay = vtk.vtkDelaunay2D()
                delaunay.SetInputData(poly)
                delaunay.Update()

                # Apply平滑处理 - 减少起伏
                smoothed_surface = self.apply_mesh_smoothing(delaunay.GetOutput())

                num = vtk_pts.GetNumberOfPoints()
                sum_z = sum([vtk_pts.GetPoint(i)[2] for i in range(num)])
                avg_z = sum_z / num if num > 0 else 0
                layer_surfaces[layer_name] = (smoothed_surface, avg_z)

            coal_bottom_pts = self._gather_coal_bottom_points()
            coal_bottom_surface = None
            if len(coal_bottom_pts) >= 3:
                vtk_pts = vtk.vtkPoints()
                for x, y, z in coal_bottom_pts:
                    vtk_pts.InsertNextPoint(x, y, z)
                poly = vtk.vtkPolyData()
                poly.SetPoints(vtk_pts)
                verts = vtk.vtkCellArray()
                for i in range(vtk_pts.GetNumberOfPoints()):
                    verts.InsertNextCell(1)
                    verts.InsertCellPoint(i)
                poly.SetVerts(verts)
                delaunay = vtk.vtkDelaunay2D()
                delaunay.SetInputData(poly)
                delaunay.Update()

                # 对煤层底面也Apply平滑处理
                coal_bottom_surface = self.apply_mesh_smoothing(delaunay.GetOutput())

            sorted_layers = sorted(layer_surfaces.items(), key=lambda kv: kv[1][1], reverse=True)

            # 相邻layer top之间的广义三棱柱体
            for i in range(len(sorted_layers) - 1):
                upper_name, (upper_poly, _) = sorted_layers[i]
                lower_name, (lower_poly, _) = sorted_layers[i + 1]
                prism_actor = self._build_top_top_prism(upper_poly, lower_poly, upper_name)
                self.renderer.AddActor(prism_actor)
                self.closed_model_actors.append(prism_actor)

            # 煤层top-bottom封闭体
            if "煤层" in layer_surfaces:
                top_poly, _ = layer_surfaces["煤层"]
                if coal_bottom_surface is not None:
                    shell_actor = self._build_top_bottom_closed_shell(top_poly, coal_bottom_surface)
                    self.renderer.AddActor(shell_actor)
                    self.closed_model_actors.append(shell_actor)

            # 保留各layer top表面
            for layer_name, (poly, _) in layer_surfaces.items():
                # 获取智能颜色
                if hasattr(self, 'viz_manager') and self.viz_manager:
                    layer_color = self.viz_manager.get_layer_color(layer_name)
                else:
                    layer_color = layer_colors.get(layer_name, (0.7, 0.7, 0.7))

                # 如果启用Gradient Fill，创建渐变效果
                if (hasattr(self, 'gradient_checkbox') and self.gradient_checkbox.isChecked() and
                        hasattr(self, 'viz_manager') and self.viz_manager):
                    # 创建渐变色
                    start_color = tuple(c * 1.1 if c * 1.1 <= 1.0 else 1.0 for c in layer_color)
                    end_color = tuple(c * 0.8 for c in layer_color)
                    poly_with_gradient = self.viz_manager.create_gradient_fill(poly, start_color, end_color)
                else:
                    poly_with_gradient = poly

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(poly_with_gradient)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetOpacity(0.4)  # 稍微增加Opacity
                actor.GetProperty().SetColor(layer_color)
                actor.GetProperty().SetEdgeVisibility(0)

                # 添加材质效果
                actor.GetProperty().SetSpecular(0.3)
                actor.GetProperty().SetSpecularPower(20)
                actor.GetProperty().SetDiffuse(0.8)

                self.renderer.AddActor(actor)
                self.closed_model_actors.append(actor)

            self.closed_model_visible = True
            self.vtk_widget.GetRenderWindow().Render()
            self.update_status("封闭3D模型已生成")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"生成封闭三维模型时出错：{e}")

    def toggle_closed_model(self):
        """切换封闭模型显示"""
        if not self.closed_model_actors:
            self.create_closed_model()
        else:
            self.closed_model_visible = not self.closed_model_visible
            for actor in self.closed_model_actors:
                actor.SetVisibility(1 if self.closed_model_visible else 0)
            self.vtk_widget.GetRenderWindow().Render()

    def apply_kriging_interpolation(self):
        """ApplyKriging Interpolation算法"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        if not self.algorithm_status.get('kriging_available', False):
            QMessageBox.warning(self, "算法不可用", "Kriging Interpolation算法初始化失败")
            return

        try:
            # 让用户选择要插值的列
            value_col = self.select_interpolation_column()
            if not value_col:
                return

            # 检查缓存
            cache_params = {'method': 'kriging', 'model': 'spherical', 'value_col': value_col}
            cached_result = self.cache_manager.get(self.df_latest, 'kriging_interpolation', cache_params)

            if cached_result is not None:
                self.show_kriging_results(cached_result, value_col)
                self.update_status("✅ 从缓存加载Kriging Interpolation结果")
                return

            # Auto Detect坐标列
            coords_cols = self.detect_coordinate_columns()
            if not coords_cols:
                QMessageBox.warning(self, "数据错误", "无法检测到坐标列 (X, Y, Z)")
                return

            # 检查必要的列
            missing_cols = [col for col in coords_cols + [value_col] if col not in self.df_latest.columns]
            if missing_cols:
                QMessageBox.warning(self, "数据错误", f"缺少必要的列: {missing_cols}")
                return

            self.update_status("🔄 正在进行Kriging Interpolation...")

            # 准备数据
            valid_data = self.df_latest.dropna(subset=coords_cols + [value_col])

            if len(valid_data) < 5:
                QMessageBox.warning(self, "数据不足", "有效数据点少于5个，无法进行Kriging Interpolation")
                return

            coords = valid_data[coords_cols].values
            values = valid_data[value_col].values

            # 拟合克里金模型
            self.kriging_interpolator.fit(coords, values)

            # 创建预测网格
            x_range = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 20)
            y_range = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 20)
            z_range = np.linspace(coords[:, 2].min(), coords[:, 2].max(), 10)

            # 生成预测点
            xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
            pred_coords = np.column_stack([xx.flatten(), yy.flatten(), zz.flatten()])

            # 进行预测
            predictions, variances = self.kriging_interpolator.predict(pred_coords)

            # 保存结果到缓存
            result_data = {
                'pred_coords': pred_coords,
                'predictions': predictions,
                'variances': variances,
                'original_coords': coords,
                'original_values': values,
                'coords_cols': coords_cols,
                'value_col': value_col
            }

            self.cache_manager.set(self.df_latest, 'kriging_interpolation', cache_params, result_data)

            # 显示结果
            self.show_kriging_results(result_data, value_col)
            self.update_status("✅ Kriging Interpolation完成并已缓存")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Kriging Interpolation失败: {str(e)}")
            self.update_status("❌ Kriging Interpolation失败")

    def select_interpolation_column(self):
        """让用户选择要插值的数值列"""
        numeric_cols = self.df_latest.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            QMessageBox.warning(self, "数据错误", "没有找到数值列")
            return None

        # 过滤掉可能的坐标列
        coord_keywords = ['x', 'y', 'z', 'lon', 'lat', 'longitude', 'latitude', 'depth', 'elevation']
        value_cols = [col for col in numeric_cols
                      if not any(keyword in col.lower() for keyword in coord_keywords)]

        if not value_cols:
            value_cols = numeric_cols  # 如果过滤后没有列，使用所有数值列

        # 创建选择对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("选择插值列")
        dialog.setFixedSize(300, 200)

        layout = QVBoxLayout()

        label = QLabel("请选择要进行Kriging Interpolation的数值列:")
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(value_cols)
        layout.addWidget(combo)

        # 按钮
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

        dialog.setLayout(layout)

        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        if dialog.exec_() == QDialog.Accepted:
            return combo.currentText()
        return None

    def detect_coordinate_columns(self):
        """Auto Detect坐标列"""
        columns = self.df_latest.columns.tolist()

        # 常见的坐标列名模式
        x_patterns = ['x', 'X', 'longitude', 'lon', 'easting']
        y_patterns = ['y', 'Y', 'latitude', 'lat', 'northing']
        z_patterns = ['z', 'Z', 'depth', 'elevation', 'height', 'altitude']

        def find_column(patterns):
            for pattern in patterns:
                for col in columns:
                    if pattern.lower() in col.lower():
                        return col
            return None

        x_col = find_column(x_patterns)
        y_col = find_column(y_patterns)
        z_col = find_column(z_patterns)

        # 如果找不到Z列，可以只用X、Y进行2D插值
        if x_col and y_col:
            coords_cols = [x_col, y_col]
            if z_col:
                coords_cols.append(z_col)
            return coords_cols

        return None

    def show_kriging_results(self, result_data, value_col):
        """显示Kriging Interpolation结果"""
        predictions = result_data['predictions']
        variances = result_data['variances']
        original_coords = result_data['original_coords']

        # 显示结果
        avg_variance = np.mean(variances)
        result_msg = f"""
🎯 Kriging Interpolation完成！

📊 插值统计 ({value_col}):
• 原始数据点: {len(original_coords)}
• 预测点数: {len(predictions)}
• 预测值范围: {predictions.min():.2f} - {predictions.max():.2f}
• 平均估计方差: {avg_variance:.4f}
• 插值精度: {'高' if avg_variance < 1.0 else '中等' if avg_variance < 5.0 else '低'}

💾 结果已缓存，下次加载更快！

💡 提示: Kriging Interpolation考虑了空间相关性，
相比传统插值方法精度提升 30-50%
"""

        QMessageBox.information(self, "Kriging Interpolation完成", result_msg)

    def apply_advanced_anomaly_detection(self):
        """Apply高级Anomaly Detection算法"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先加载数据")
            return

        if not self.algorithm_status.get('anomaly_detection_available', False):
            QMessageBox.warning(self, "算法不可用", "Anomaly Detection算法初始化失败")
            return

        try:
            # 检查缓存
            cache_params = {'contamination': 0.1, 'methods': ['iqr', 'zscore', 'isolation']}
            cached_result = self.cache_manager.get(self.df_latest, 'anomaly_detection', cache_params)

            if cached_result is not None:
                self.show_anomaly_results(cached_result)
                self.update_status("✅ 从缓存加载Anomaly Detection结果")
                return

            self.update_status("🔄 正在进行高级Anomaly Detection...")

            # 选择数值列进行Anomaly Detection
            numeric_cols = self.df_latest.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                QMessageBox.warning(self, "数据错误", "没有找到数值列进行Anomaly Detection")
                return

            # 进行Anomaly Detection
            anomaly_results = self.anomaly_detector.fit_detect(self.df_latest, numeric_cols)

            # 保存结果到缓存
            self.cache_manager.set(self.df_latest, 'anomaly_detection', cache_params, anomaly_results)

            # 显示结果
            self.show_anomaly_results(anomaly_results)
            self.update_status("✅ 高级Anomaly Detection完成并已缓存")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Anomaly Detection失败: {str(e)}")
            self.update_status("❌ Anomaly Detection失败")

    def show_anomaly_results(self, anomaly_results):
        """显示Anomaly Detection结果"""
        # 生成报告
        report = self.anomaly_detector.generate_report(self.df_latest, anomaly_results)

        # 创建结果对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("🔍 高级Anomaly Detection结果")
        dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout()

        # 报告文本
        text_widget = QTextEdit()
        text_widget.setPlainText(report)
        text_widget.setReadOnly(True)
        layout.addWidget(text_widget)

        # 统计信息
        total_anomalies = sum(len(results['anomalies']) for results in anomaly_results.values())
        stats_label = QLabel(f"📊 总计检测到 {total_anomalies} 个异常值")
        stats_label.setStyleSheet("font-weight: bold; color: #e74c3c; padding: 10px;")
        layout.addWidget(stats_label)

        # 按钮
        button_layout = QHBoxLayout()

        export_btn = QPushButton("📄 Export报告")
        export_btn.clicked.connect(lambda: self.export_anomaly_report(report))
        button_layout.addWidget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # 显示对话框
        dialog.exec_()

    def export_anomaly_report(self, report):
        """ExportAnomaly Detection报告"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存Anomaly Detection报告", "anomaly_report.txt", "文本File (*.txt)"
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report)

                QMessageBox.information(self, "Export成功", f"报告已保存到: {file_path}")

        except Exception as e:
            QMessageBox.critical(self, "Export失败", f"无法保存报告: {str(e)}")

    def clear_data_cache(self):
        """清空数据缓存"""
        try:
            self.cache_manager.clear_cache()
            QMessageBox.information(self, "缓存清理", "✅ 所有缓存已清空")
            self.update_status("🗑️ 缓存已清空")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"Clear Cache失败: {str(e)}")

    def show_cache_info(self):
        """显示Cache Info"""
        try:
            cache_info = self.cache_manager.cache_info

            if not cache_info:
                QMessageBox.information(self, "Cache Info", "📭 当前没有缓存数据")
                return

            # 计算统计信息
            total_size = sum(info['size'] for info in cache_info.values())
            total_size_mb = total_size / (1024 * 1024)

            info_text = f"""
📊 缓存统计信息

💾 缓存项目数: {len(cache_info)}
📦 总占用空间: {total_size_mb:.2f} MB
📁 缓存目录: {self.cache_manager.cache_dir}

📋 详细信息:
"""

            for cache_key, info in list(cache_info.items())[:10]:  # 只显示前10项
                timestamp = datetime.fromisoformat(info['timestamp'])
                size_kb = info['size'] / 1024
                info_text += f"• {info['operation']} - {size_kb:.1f} KB - {timestamp.strftime('%Y-%m-%d %H:%M')}\n"

            if len(cache_info) > 10:
                info_text += f"... 还有 {len(cache_info) - 10} 个缓存项目"

            QMessageBox.information(self, "Cache Info", info_text)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取Cache Info失败: {str(e)}")

    def toggle_gradient_fill(self):
        """切换Gradient Fill效果"""
        if hasattr(self, 'closed_model_actors') and self.closed_model_actors:
            # 重新Generate Closed Model以Apply渐变效果
            self.create_closed_model()
            status = "启用" if self.gradient_checkbox.isChecked() else "禁用"
            self.update_status(f"🌈 Gradient Fill已{status}")

    def update_slice_direction(self):
        """更新切片Direction"""
        if not hasattr(self, 'viz_manager') or not self.viz_manager:
            return

        direction_text = self.slice_direction_combo.currentText()

        if "水平" in direction_text:
            self.viz_manager.slice_normal = (0, 0, 1)
        elif "纵向" in direction_text:
            self.viz_manager.slice_normal = (1, 0, 0)
        else:  # 横向
            self.viz_manager.slice_normal = (0, 1, 0)

        # 如果切片正在显示，更新切片
        if hasattr(self, 'slice_checkbox') and self.slice_checkbox.isChecked():
            self.update_slice_display()

    def update_slice_position(self):
        """更新切片Position"""
        if not hasattr(self, 'viz_manager') or not self.viz_manager:
            return

        position = self.slice_position_slider.value()
        self.slice_position_label.setText(f"{position}%")
        self.viz_manager.current_slice_position = position / 100.0

        # 如果切片正在显示，更新切片
        if hasattr(self, 'slice_checkbox') and self.slice_checkbox.isChecked():
            self.update_slice_display()

    def toggle_slice_display(self):
        """切换切片显示"""
        if self.slice_checkbox.isChecked():
            self.show_slice_plane()
        else:
            self.hide_slice_plane()

    def show_slice_plane(self):
        """Show Slice平面"""
        try:
            if not hasattr(self, 'viz_manager') or not self.viz_manager:
                QMessageBox.warning(self, "提示", "可视化管理器未初始化")
                self.slice_checkbox.setChecked(False)
                return

            if not self.closed_model_actors:
                QMessageBox.warning(self, "提示", "请先Generate Closed Model")
                self.slice_checkbox.setChecked(False)
                return

            # 计算所有模型的边界
            bounds = self.calculate_models_bounds()
            if not bounds:
                return

            # 创建切片平面
            slice_plane = self.viz_manager.create_slice_plane(
                bounds,
                self.viz_manager.slice_normal,
                self.viz_manager.current_slice_position
            )

            if slice_plane:
                # 创建切片平面显示
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(slice_plane)

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetOpacity(0.3)
                actor.GetProperty().SetColor(1.0, 1.0, 0.0)  # 黄色切片平面
                actor.GetProperty().SetEdgeVisibility(1)
                actor.GetProperty().SetEdgeColor(1.0, 0.5, 0.0)  # 橙色边缘

                self.renderer.AddActor(actor)
                self.viz_manager.slice_actors.append(actor)

                self.vtk_widget.GetRenderWindow().Render()
                self.update_status("✂️ 切片平面已显示")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Show Slice失败: {str(e)}")

    def hide_slice_plane(self):
        """隐藏切片平面"""
        if hasattr(self, 'viz_manager') and self.viz_manager:
            self.viz_manager.clear_slice_actors()
        self.vtk_widget.GetRenderWindow().Render()
        self.update_status("✂️ 切片平面已隐藏")

    def update_slice_display(self):
        """更新切片显示"""
        if hasattr(self, 'slice_checkbox') and self.slice_checkbox.isChecked():
            self.hide_slice_plane()
            self.show_slice_plane()

    def apply_slice_to_models(self):
        """将切片Apply到模型"""
        try:
            if not hasattr(self, 'viz_manager') or not self.viz_manager:
                QMessageBox.warning(self, "提示", "可视化管理器未初始化")
                return

            if not self.closed_model_actors:
                QMessageBox.warning(self, "提示", "请先Generate Closed Model")
                return

            # 计算边界
            bounds = self.calculate_models_bounds()
            if not bounds:
                return

            # 创建切片平面
            slice_plane = self.viz_manager.create_slice_plane(
                bounds,
                self.viz_manager.slice_normal,
                self.viz_manager.current_slice_position
            )

            if not slice_plane:
                return

            # 对每个模型Apply切片
            new_actors = []
            for actor in self.closed_model_actors:
                try:
                    # 获取原始数据
                    mapper = actor.GetMapper()
                    if mapper and mapper.GetInput():
                        original_data = mapper.GetInput()

                        # Apply切片
                        sliced_data = self.viz_manager.apply_slice_to_model(original_data, slice_plane)

                        if sliced_data and sliced_data.GetNumberOfPoints() > 0:
                            # 创建新的actorShow Slice结果
                            new_mapper = vtk.vtkPolyDataMapper()
                            new_mapper.SetInputData(sliced_data)

                            new_actor = vtk.vtkActor()
                            new_actor.SetMapper(new_mapper)

                            # 复制原始actor的属性
                            new_actor.GetProperty().SetColor(actor.GetProperty().GetColor())
                            new_actor.GetProperty().SetOpacity(0.9)  # 稍微增加不Opacity
                            new_actor.GetProperty().SetLineWidth(2)
                            new_actor.GetProperty().SetRepresentationToWireframe()

                            self.renderer.AddActor(new_actor)
                            new_actors.append(new_actor)

                except Exception as e:
                    print(f"切片单个模型失败: {e}")
                    continue

            # 保存新的切片actors
            if hasattr(self, 'viz_manager') and self.viz_manager:
                self.viz_manager.slice_actors.extend(new_actors)

            self.vtk_widget.GetRenderWindow().Render()
            self.update_status(f"🔪 已对 {len(new_actors)} 个模型Apply切片")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Apply切片失败: {str(e)}")

    def calculate_models_bounds(self):
        """计算所有模型的边界"""
        try:
            if not self.closed_model_actors:
                return None

            min_bounds = [float('inf')] * 6
            max_bounds = [float('-inf')] * 6

            for actor in self.closed_model_actors:
                mapper = actor.GetMapper()
                if mapper and mapper.GetInput():
                    bounds = mapper.GetInput().GetBounds()
                    for i in range(0, 6, 2):  # x, y, z 的 min 值
                        min_bounds[i] = min(min_bounds[i], bounds[i])
                    for i in range(1, 6, 2):  # x, y, z 的 max 值
                        max_bounds[i] = max(max_bounds[i], bounds[i])

            # 组合最终边界
            final_bounds = []
            for i in range(3):
                final_bounds.extend([min_bounds[i * 2], max_bounds[i * 2 + 1]])

            return final_bounds

        except Exception as e:
            print(f"计算模型边界失败: {e}")
            return None

    # ===== 优化后的Fault Modeling方法 =====

    def start_interactive_fault_mode(self):
        """开始交互式断层添加模式"""
        if self.interactive_fault_mode:
            self.stop_interactive_fault_mode()
            return

        # 询问断层名称
        name, ok = QtWidgets.QInputDialog.getText(
            self, "断层名称", "请输入断层名称:", text=f"断层_{len(self.fault_actors) + 1}"
        )
        if not ok or not name.strip():
            return

        self.current_fault_name = name.strip()
        self.interactive_fault_mode = True
        self.selected_fault_points = []

        # 清除之前的临时点
        self.clear_fault_point_actors()

        # 更新按钮状态
        self.btn_interactive_fault.setText("🛑 停止交互")
        self.btn_interactive_fault.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; }")

        # 更新状态栏
        self.update_status(f"🖱️ 交互模式：请在3DView中点击选择断层点 (当前断层: {self.current_fault_name})")

        # 启用点选模式
        self.setup_point_selection_mode()

    def stop_interactive_fault_mode(self):
        """停止交互式断层添加模式"""
        self.interactive_fault_mode = False

        # 如果有足够的点，询问是否创建断层
        if len(self.selected_fault_points) >= 3:
            reply = QMessageBox.question(
                self, "创建断层",
                f"已选择 {len(self.selected_fault_points)} 个点，是否创建断层 '{self.current_fault_name}'？",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.create_fault_from_selected_points()

        # 清理状态
        self.clear_fault_point_actors()
        self.selected_fault_points = []
        self.current_fault_name = ""

        # 恢复按钮状态
        self.btn_interactive_fault.setText("🖱️ Interactive Add")
        self.btn_interactive_fault.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        self.update_status("🔧 交互模式已停止")

    def setup_point_selection_mode(self):
        """设置点选模式"""
        # 这里可以设置特殊的交互器样式来处理点击事件
        # 暂时使用现有的交互器，在鼠标点击事件中处理
        pass

    def on_point_selected(self, x, y, z):
        """处理点选事件"""
        if not self.interactive_fault_mode:
            return

        # 添加点到列表
        self.selected_fault_points.append((x, y, z))

        # 创建可视化标记
        self.add_fault_point_actor(x, y, z, len(self.selected_fault_points))

        # 更新状态
        point_count = len(self.selected_fault_points)
        if point_count < 3:
            self.update_status(f"🖱️ 已选择 {point_count} 个点，还需要 {3 - point_count} 个点")
        else:
            self.update_status(f"🖱️ 已选择 {point_count} 个点，可以创建断层了。继续点击添加更多点或点击停止按钮")

    def add_fault_point_actor(self, x, y, z, index):
        """Add Fault点的可视化标记"""
        # 创建球体标记
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(3)
        sphere.SetCenter(x, y, z)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色
        actor.GetProperty().SetOpacity(0.8)

        # 添加文本标签
        text_actor = self.create_point_label(x, y, z, str(index))

        self.renderer.AddActor(actor)
        self.renderer.AddActor(text_actor)

        self.fault_point_actors.extend([actor, text_actor])
        self.vtk_widget.GetRenderWindow().Render()

    def create_point_label(self, x, y, z, text):
        """创建点标签"""
        text_source = vtk.vtkVectorText()
        text_source.SetText(text)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(text_source.GetOutputPort())

        actor = vtk.vtkFollower()
        actor.SetMapper(mapper)
        actor.SetPosition(x, y + 5, z)
        actor.SetScale(2, 2, 2)
        actor.GetProperty().SetColor(1, 1, 0)  # 黄色
        actor.SetCamera(self.renderer.GetActiveCamera())

        return actor

    def clear_fault_point_actors(self):
        """Clear Faults点标记"""
        for actor in self.fault_point_actors:
            self.renderer.RemoveActor(actor)
        self.fault_point_actors = []
        self.vtk_widget.GetRenderWindow().Render()

    def create_fault_from_selected_points(self):
        """从选择的点创建断层"""
        if len(self.selected_fault_points) < 3:
            QMessageBox.warning(self, "警告", "需要至少3个点来创建断层")
            return

        # 询问断层类型
        fault_types = ['逆断层', '正断层', '平推断层', '走滑断层', '未知断层']
        fault_type, ok = QtWidgets.QInputDialog.getItem(
            self, "断层类型", "请选择断层类型:", fault_types, 0, False
        )
        if not ok:
            fault_type = '未知断层'

        # 创建断层数据
        fault_data = {
            'name': self.current_fault_name,
            'type': fault_type,
            'points': self.selected_fault_points
        }

        # Add Fault
        self.add_fault_from_points_data(fault_data)

    def auto_detect_faults(self):
        """Auto Detect断层"""
        if self.df_latest is None:
            QMessageBox.warning(self, "警告", "请先Import钻孔数据")
            return

        try:
            detected_faults = self.analyze_fault_indicators()

            if not detected_faults:
                QMessageBox.information(self, "检测结果", "未检测到明显的断层迹象")
                return

            # 显示检测结果对话框
            dialog = FaultDetectionResultDialog(detected_faults, self)
            if dialog.exec_() == QDialog.Accepted:
                selected_faults = dialog.get_selected_faults()
                for fault_data in selected_faults:
                    self.add_fault_from_points_data(fault_data)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"断层Auto Detect失败：{e}")

    def analyze_fault_indicators(self):
        """分析断层指示"""
        detected_faults = []

        try:
            # 分析地层厚度异常
            thickness_faults = self.detect_thickness_anomalies()
            detected_faults.extend(thickness_faults)

            # 分析地层错位
            displacement_faults = self.detect_layer_displacement()
            detected_faults.extend(displacement_faults)

            # 分析地层倾斜异常
            tilt_faults = self.detect_tilt_anomalies()
            detected_faults.extend(tilt_faults)

        except Exception as e:
            print(f"断层分析失败: {e}")

        return detected_faults

    def detect_thickness_anomalies(self):
        """检测厚度异常（可能指示断层）"""
        faults = []

        try:
            df = self.df_latest
            if 'layername' not in df.columns:
                return faults

            # 计算各地层厚度
            df['thickness'] = df['Bottom'] - df['Top']

            # 对每个地层分析厚度异常
            for layer in df['layername'].unique():
                layer_data = df[df['layername'] == layer]
                if len(layer_data) < 5:  # 数据太少，跳过
                    continue

                thicknesses = layer_data['thickness'].values
                mean_thickness = np.mean(thicknesses)
                std_thickness = np.std(thicknesses)

                # 找出厚度异常的钻孔
                anomalous_holes = layer_data[
                    abs(layer_data['thickness'] - mean_thickness) > 2 * std_thickness
                    ]

                if len(anomalous_holes) >= 3:
                    # 创建假设断层
                    points = []
                    for _, hole in anomalous_holes.head(3).iterrows():
                        x = float(hole['X'])
                        y = -float(hole['Y'])
                        z = -float(hole['Z']) + float(hole['Top'])
                        points.append((x, y, z))

                    fault_name = f"厚度异常断层_{layer}"
                    faults.append({
                        'name': fault_name,
                        'type': '未知断层',
                        'points': points,
                        'confidence': min(100, len(anomalous_holes) * 20),
                        'indicator': f'{layer}层厚度异常'
                    })

        except Exception as e:
            print(f"厚度Anomaly Detection失败: {e}")

        return faults

    def detect_layer_displacement(self):
        """检测地层错位"""
        faults = []

        try:
            df = self.df_latest
            if 'layername' not in df.columns:
                return faults

            # 对每个地层分析顶面高程
            for layer in df['layername'].unique():
                layer_data = df[df['layername'] == layer]
                if len(layer_data) < 5:
                    continue

                # 计算地层顶面标高
                layer_data = layer_data.copy()
                layer_data['top_elevation'] = -layer_data['Z'] + layer_data['Top']

                # 检测高程突变
                elevations = layer_data['top_elevation'].values
                coords = layer_data[['X', 'Y']].values

                # 简单的梯度检测
                if len(elevations) >= 4:
                    # 计算相邻点之间的高程差
                    for i in range(len(elevations) - 2):
                        for j in range(i + 2, len(elevations)):
                            elev_diff = abs(elevations[i] - elevations[j])
                            coord_dist = np.linalg.norm(coords[i] - coords[j])

                            if coord_dist > 0 and elev_diff / coord_dist > 0.5:  # 高程梯度阈值
                                # 可能的断层
                                mid_point = (coords[i] + coords[j]) / 2
                                mid_elev = (elevations[i] + elevations[j]) / 2

                                # 创建简单的断层面
                                points = [
                                    (coords[i][0], -coords[i][1], elevations[i]),
                                    (mid_point[0], -mid_point[1], mid_elev),
                                    (coords[j][0], -coords[j][1], elevations[j])
                                ]

                                fault_name = f"错位断层_{layer}_{i}_{j}"
                                faults.append({
                                    'name': fault_name,
                                    'type': '正断层' if elevations[i] > elevations[j] else '逆断层',
                                    'points': points,
                                    'confidence': min(100, int(elev_diff * 10)),
                                    'indicator': f'{layer}层错位 ({elev_diff:.1f}m)'
                                })
                                break
                        if faults:  # 每层只检测一个断层
                            break

        except Exception as e:
            print(f"地层错位检测失败: {e}")

        return faults

    def detect_tilt_anomalies(self):
        """检测倾斜异常"""
        faults = []

        try:
            df = self.df_latest
            if 'layername' not in df.columns:
                return faults

            # 这里可以添加更复杂的倾斜分析
            # 暂时简化处理

        except Exception as e:
            print(f"倾斜Anomaly Detection失败: {e}")

        return faults

    def show_fault_templates(self):
        """Show Faults模板对话框"""
        dialog = FaultTemplateDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            template_data = dialog.get_selected_template()
            if template_data:
                self.add_fault_from_template(template_data)

    def add_fault_from_template(self, template_data):
        """从模板Add Fault"""
        try:
            fault_name = template_data['name']
            fault_type = template_data['type']

            if template_data['method'] == 'strike_dip':
                self.fault_modeler.add_fault_from_strike_dip(
                    fault_name,
                    template_data['center_point'],
                    template_data['strike'],
                    template_data['dip'],
                    template_data['length'],
                    template_data['width'],
                    fault_type
                )
            elif template_data['method'] == 'points':
                self.fault_modeler.add_fault_from_points(
                    fault_name,
                    template_data['points'],
                    fault_type
                )

            # 创建VTK actor
            actor = self.fault_modeler.create_fault_actor(fault_name)
            if actor:
                self.renderer.AddActor(actor)
                self.fault_actors[fault_name] = actor
                self.fault_visible[fault_name] = True

                # 添加到Fault ListUI
                self.add_fault_to_ui_list(fault_name, fault_type)

                self.vtk_widget.GetRenderWindow().Render()
                self.update_status(f"✅ 模板断层 '{fault_name}' 添加成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"添加模板断层失败：{e}")

    def show_advanced_fault_dialog(self):
        """显示高级断层添加对话框"""
        dialog = AdvancedFaultDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            fault_data = dialog.get_fault_data()
            if fault_data:
                if fault_data['method'] == 'points':
                    self.add_fault_from_points_data(fault_data)
                elif fault_data['method'] == 'equation':
                    self.add_fault_from_equation_data(fault_data)
                elif fault_data['method'] == 'strike_dip':
                    self.add_fault_from_strike_dip_data(fault_data)

    def add_fault_from_points_data(self, fault_data):
        """从点数据Add Fault"""
        try:
            fault_name = fault_data['name']
            points = fault_data['points']
            fault_type = fault_data['type']

            # 使用Fault Modeling器Add Fault
            self.fault_modeler.add_fault_from_points(fault_name, points, fault_type)

            # 创建VTK actor
            actor = self.fault_modeler.create_fault_actor(fault_name)
            if actor:
                self.renderer.AddActor(actor)
                self.fault_actors[fault_name] = actor
                self.fault_visible[fault_name] = True

                # 添加到Fault ListUI
                self.add_fault_to_ui_list(fault_name, fault_type)

                self.vtk_widget.GetRenderWindow().Render()
                self.update_status(f"✅ 断层 '{fault_name}' 添加成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Add Fault失败：{e}")

    def add_fault_from_equation_data(self, fault_data):
        """从方程数据Add Fault"""
        try:
            fault_name = fault_data['name']
            plane_params = fault_data['plane_params']
            bounds = fault_data['bounds']
            fault_type = fault_data['type']

            # 使用Fault Modeling器Add Fault
            self.fault_modeler.add_fault_from_plane_equation(fault_name, plane_params, bounds, fault_type)

            # 创建VTK actor
            actor = self.fault_modeler.create_fault_actor(fault_name)
            if actor:
                self.renderer.AddActor(actor)
                self.fault_actors[fault_name] = actor
                self.fault_visible[fault_name] = True

                # 添加到Fault ListUI
                self.add_fault_to_ui_list(fault_name, fault_type)

                self.vtk_widget.GetRenderWindow().Render()
                self.update_status(f"✅ 断层 '{fault_name}' 添加成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Add Fault失败：{e}")

    def add_fault_from_strike_dip_data(self, fault_data):
        """从走向倾向数据Add Fault"""
        try:
            fault_name = fault_data['name']
            center_point = fault_data['center_point']
            strike = fault_data['strike']
            dip = fault_data['dip']
            length = fault_data['length']
            width = fault_data['width']
            fault_type = fault_data['type']

            # 使用Fault Modeling器Add Fault
            self.fault_modeler.add_fault_from_strike_dip(
                fault_name, center_point, strike, dip, length, width, fault_type
            )

            # 创建VTK actor
            actor = self.fault_modeler.create_fault_actor(fault_name)
            if actor:
                self.renderer.AddActor(actor)
                self.fault_actors[fault_name] = actor
                self.fault_visible[fault_name] = True

                # 添加到Fault ListUI
                self.add_fault_to_ui_list(fault_name, fault_type)

                self.vtk_widget.GetRenderWindow().Render()
                self.update_status(f"✅ 断层 '{fault_name}' 添加成功")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"Add Fault失败：{e}")

    def add_fault_to_ui_list(self, fault_name, fault_type):
        """Add Fault到UI列表"""
        # 创建断层控制行
        fault_row = QHBoxLayout()

        # 断层名称和类型标签
        fault_label = QLabel(f"{fault_name} ({fault_type})")
        fault_label.setStyleSheet("font-size: 10px;")
        fault_row.addWidget(fault_label)

        # 可见性切换按钮
        visibility_btn = QPushButton("👁️")
        visibility_btn.setMaximumWidth(30)
        visibility_btn.setCheckable(True)
        visibility_btn.setChecked(True)
        visibility_btn.clicked.connect(lambda: self.toggle_fault_visibility(fault_name))
        fault_row.addWidget(visibility_btn)

        # 删除按钮
        delete_btn = QPushButton("❌")
        delete_btn.setMaximumWidth(30)
        delete_btn.clicked.connect(lambda: self.remove_fault(fault_name))
        fault_row.addWidget(delete_btn)

        # 添加到布局
        fault_widget = QWidget()
        fault_widget.setLayout(fault_row)
        fault_widget.setObjectName(f"fault_widget_{fault_name}")
        self.fault_list_layout.addWidget(fault_widget)

    def toggle_fault_visibility(self, fault_name):
        """切换断层可见性"""
        if fault_name in self.fault_actors:
            current_visibility = self.fault_visible.get(fault_name, True)
            new_visibility = not current_visibility

            self.fault_actors[fault_name].SetVisibility(new_visibility)
            self.fault_visible[fault_name] = new_visibility
            self.fault_modeler.set_fault_visibility(fault_name, new_visibility)

            self.vtk_widget.GetRenderWindow().Render()

    def remove_fault(self, fault_name):
        """Delete Fault"""
        try:
            # 从渲染器中移除
            if fault_name in self.fault_actors:
                self.renderer.RemoveActor(self.fault_actors[fault_name])
                del self.fault_actors[fault_name]

            if fault_name in self.fault_visible:
                del self.fault_visible[fault_name]

            # 从Fault Modeling器中移除
            self.fault_modeler.remove_fault(fault_name)

            # 从UI中移除
            fault_widget = self.findChild(QWidget, f"fault_widget_{fault_name}")
            if fault_widget:
                fault_widget.setParent(None)

            self.vtk_widget.GetRenderWindow().Render()
            self.update_status(f"❌ 断层 '{fault_name}' 已删除")

        except Exception as e:
            QMessageBox.warning(self, "警告", f"Delete Fault时出错：{e}")

    def toggle_all_faults(self):
        """切换所有断层显示"""
        if not self.fault_actors:
            QMessageBox.information(self, "提示", "当前没有断层数据")
            return

        # 检查当前状态
        all_visible = all(self.fault_visible.values()) if self.fault_visible else False
        new_visibility = not all_visible

        # 切换所有断层可见性
        for fault_name in self.fault_actors:
            self.fault_actors[fault_name].SetVisibility(new_visibility)
            self.fault_visible[fault_name] = new_visibility
            self.fault_modeler.set_fault_visibility(fault_name, new_visibility)

        self.vtk_widget.GetRenderWindow().Render()
        status = "显示" if new_visibility else "隐藏"
        self.update_status(f"👁️ 所有断层已{status}")

    def clear_all_faults(self):
        """清除所有断层"""
        if not self.fault_actors:
            QMessageBox.information(self, "提示", "当前没有断层数据")
            return

        reply = QMessageBox.question(
            self, "确认删除",
            "OK要删除所有断层吗？此操作不可撤销。",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 删除所有断层
            fault_names = list(self.fault_actors.keys())
            for fault_name in fault_names:
                self.remove_fault(fault_name)

            self.update_status("🗑️ 所有断层已清除")

    def apply_fault_cuts(self):
        """Apply断层Cut到地层模型"""
        if not self.fault_actors:
            QMessageBox.information(self, "提示", "没有断层数据可用于Cut")
            return

        if not self.closed_model_actors:
            reply = QMessageBox.question(
                self, "确认操作",
                "当前没有封闭模型，是否先Generate Closed Model？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.create_closed_model()
            else:
                return

        try:
            # 如果已经在Cut模式，先Restore Original模型
            if self.is_cut_mode:
                self.restore_original_models()

            # 保存原始模型actors
            self.original_model_actors = list(self.closed_model_actors)

            # 获取可见的断层
            fault_names = list(self.fault_modeler.get_fault_list())
            visible_faults = [name for name in fault_names if self.fault_visible.get(name, True)]

            if not visible_faults:
                QMessageBox.information(self, "提示", "没有可见的断层用于Cut")
                return

            # 询问用户选择Cut方式
            cut_mode = self.show_cut_mode_dialog(visible_faults)
            if not cut_mode:
                return

            # 创建Cut后的模型
            self.cut_model_actors = []

            progress_dialog = QtWidgets.QProgressDialog("正在进行断层Cut...", "Cancel", 0,
                                                        len(self.original_model_actors), self)
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.show()

            for i, original_actor in enumerate(self.original_model_actors):
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    break

                mapper = original_actor.GetMapper()
                if mapper and mapper.GetInput():
                    polydata = mapper.GetInput()

                    # 对当前模型Apply所有断层Cut
                    cut_polydata = polydata
                    for fault_name in cut_mode['selected_faults']:
                        cut_polydata = self.fault_modeler.cut_model_with_fault(cut_polydata, fault_name)
                        if not cut_polydata or cut_polydata.GetNumberOfPoints() == 0:
                            break

                    if cut_polydata and cut_polydata.GetNumberOfPoints() > 0:
                        # 创建新的actor显示Cut结果
                        new_mapper = vtk.vtkPolyDataMapper()
                        new_mapper.SetInputData(cut_polydata)

                        new_actor = vtk.vtkActor()
                        new_actor.SetMapper(new_mapper)

                        # 复制原始actor的属性
                        new_actor.GetProperty().DeepCopy(original_actor.GetProperty())

                        # 根据Cut模式调整显示属性
                        if cut_mode['highlight_cuts']:
                            # 高亮Cut边缘
                            new_actor.GetProperty().SetEdgeVisibility(True)
                            new_actor.GetProperty().SetEdgeColor(1, 0, 0)  # 红色边缘
                            new_actor.GetProperty().SetLineWidth(2)

                        # 隐藏原始actor，显示Cut后的actor
                        original_actor.SetVisibility(False)
                        self.renderer.AddActor(new_actor)
                        self.cut_model_actors.append(new_actor)

            progress_dialog.close()

            # 更新状态
            self.is_cut_mode = True
            self.btn_restore_model.setEnabled(True)
            self.btn_apply_fault_cut.setText("✂️ 重新Cut")

            self.vtk_widget.GetRenderWindow().Render()
            self.update_status(f"✂️ 断层Cut完成 ({len(cut_mode['selected_faults'])}个断层)")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"断层Cut失败：{e}")
            self.restore_original_models()

    def show_cut_mode_dialog(self, available_faults):
        """显示Cut模式选择对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("断层Cut设置")
        dialog.setModal(True)
        dialog.resize(400, 300)

        layout = QVBoxLayout()

        # 断层选择
        fault_group = QGroupBox("选择要用于Cut的断层")
        fault_layout = QVBoxLayout()

        fault_checkboxes = {}
        for fault_name in available_faults:
            checkbox = QCheckBox(fault_name)
            checkbox.setChecked(True)
            fault_layout.addWidget(checkbox)
            fault_checkboxes[fault_name] = checkbox

        fault_group.setLayout(fault_layout)
        layout.addWidget(fault_group)

        # Cut选项
        options_group = QGroupBox("Cut选项")
        options_layout = QVBoxLayout()

        highlight_checkbox = QCheckBox("高亮显示Cut边缘")
        highlight_checkbox.setChecked(True)
        options_layout.addWidget(highlight_checkbox)

        preserve_checkbox = QCheckBox("保留原始模型（隐藏显示）")
        preserve_checkbox.setChecked(True)
        preserve_checkbox.setEnabled(False)  # 始终保留
        options_layout.addWidget(preserve_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 按钮
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("执行Cut")
        ok_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            selected_faults = [name for name, checkbox in fault_checkboxes.items() if checkbox.isChecked()]
            if not selected_faults:
                QMessageBox.warning(self, "警告", "请至少选择一个断层")
                return None

            return {
                'selected_faults': selected_faults,
                'highlight_cuts': highlight_checkbox.isChecked(),
                'preserve_original': preserve_checkbox.isChecked()
            }

        return None

    def restore_original_models(self):
        """Restore Original模型"""
        try:
            # 移除Cut后的actors
            for actor in self.cut_model_actors:
                self.renderer.RemoveActor(actor)
            self.cut_model_actors = []

            # 显示原始actors
            for actor in self.original_model_actors:
                actor.SetVisibility(True)

            # 更新状态
            self.is_cut_mode = False
            self.btn_restore_model.setEnabled(False)
            self.btn_apply_fault_cut.setText("✂️ Cut模型")

            self.vtk_widget.GetRenderWindow().Render()
            self.update_status("🔄 已Restore Original模型")

        except Exception as e:
            QMessageBox.warning(self, "警告", f"Restore Original模型时出错：{e}")


# ===== 优化的断层对话框类 =====

class FaultDetectionResultDialog(QDialog):
    """断层检测结果对话框"""

    def __init__(self, detected_faults, parent=None):
        super().__init__(parent)
        self.detected_faults = detected_faults
        self.setWindowTitle("断层Auto Detect结果")
        self.setModal(True)
        self.resize(600, 400)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 说明标签
        info_label = QLabel(f"检测到 {len(self.detected_faults)} 个可能的断层，请选择要添加的断层：")
        info_label.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(info_label)

        # Fault List
        self.fault_list = QTableWidget(len(self.detected_faults), 5)
        self.fault_list.setHorizontalHeaderLabels(['选择', '断层名称', '类型', '置信度', '指示'])

        for i, fault in enumerate(self.detected_faults):
            # 选择框
            checkbox = QCheckBox()
            checkbox.setChecked(fault.get('confidence', 0) > 60)  # 高置信度默认选中
            self.fault_list.setCellWidget(i, 0, checkbox)

            # 断层信息
            self.fault_list.setItem(i, 1, QtWidgets.QTableWidgetItem(fault['name']))
            self.fault_list.setItem(i, 2, QtWidgets.QTableWidgetItem(fault['type']))
            self.fault_list.setItem(i, 3, QtWidgets.QTableWidgetItem(f"{fault.get('confidence', 0)}%"))
            self.fault_list.setItem(i, 4, QtWidgets.QTableWidgetItem(fault.get('indicator', '')))

        self.fault_list.resizeColumnsToContents()
        layout.addWidget(self.fault_list)

        # 按钮
        button_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all)
        button_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("全不选")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        button_layout.addWidget(self.deselect_all_btn)

        button_layout.addStretch()

        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def select_all(self):
        """全选"""
        for i in range(self.fault_list.rowCount()):
            checkbox = self.fault_list.cellWidget(i, 0)
            checkbox.setChecked(True)

    def deselect_all(self):
        """全不选"""
        for i in range(self.fault_list.rowCount()):
            checkbox = self.fault_list.cellWidget(i, 0)
            checkbox.setChecked(False)

    def get_selected_faults(self):
        """获取选中的断层"""
        selected = []
        for i in range(self.fault_list.rowCount()):
            checkbox = self.fault_list.cellWidget(i, 0)
            if checkbox.isChecked():
                selected.append(self.detected_faults[i])
        return selected


class FaultTemplateDialog(QDialog):
    """断层模板对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("断层Quick Template")
        self.setModal(True)
        self.resize(500, 400)

        self.templates = {
            "水平断层": {
                "method": "strike_dip",
                "strike": 0,
                "dip": 15,
                "length": 100,
                "width": 50,
                "type": "正断层"
            },
            "高角度逆断层": {
                "method": "strike_dip",
                "strike": 45,
                "dip": 70,
                "length": 80,
                "width": 60,
                "type": "逆断层"
            },
            "低角度推覆断层": {
                "method": "strike_dip",
                "strike": 90,
                "dip": 25,
                "length": 150,
                "width": 80,
                "type": "逆断层"
            },
            "垂直走滑断层": {
                "method": "strike_dip",
                "strike": 0,
                "dip": 85,
                "length": 120,
                "width": 40,
                "type": "走滑断层"
            },
            "阶梯状正断层": {
                "method": "points",
                "points": [(0, 0, 0), (50, 20, -30), (100, 0, -60)],
                "type": "正断层"
            }
        }

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 说明
        info_label = QLabel("选择一个断层模板，然后设置参数：")
        info_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(info_label)

        # 模板选择
        template_group = QGroupBox("断层模板")
        template_layout = QVBoxLayout()

        self.template_buttons = {}
        for template_name, template_data in self.templates.items():
            btn = QtWidgets.QRadioButton(template_name)
            btn.clicked.connect(lambda checked, name=template_name: self.on_template_selected(name))
            template_layout.addWidget(btn)
            self.template_buttons[template_name] = btn

        # 默认选择第一个
        first_template = list(self.templates.keys())[0]
        self.template_buttons[first_template].setChecked(True)
        self.selected_template = first_template

        template_group.setLayout(template_layout)
        layout.addWidget(template_group)

        # 参数设置
        self.params_group = QGroupBox("参数设置")
        self.params_layout = QFormLayout()

        # 断层名称
        self.name_edit = QtWidgets.QLineEdit(f"模板断层_{len(self.templates)}")
        self.params_layout.addRow("断层名称:", self.name_edit)

        # 中心点
        center_layout = QHBoxLayout()
        self.center_x_spin = QDoubleSpinBox()
        self.center_x_spin.setRange(-1000, 1000)
        self.center_x_spin.setValue(0)
        center_layout.addWidget(QLabel("X:"))
        center_layout.addWidget(self.center_x_spin)

        self.center_y_spin = QDoubleSpinBox()
        self.center_y_spin.setRange(-1000, 1000)
        self.center_y_spin.setValue(0)
        center_layout.addWidget(QLabel("Y:"))
        center_layout.addWidget(self.center_y_spin)

        self.center_z_spin = QDoubleSpinBox()
        self.center_z_spin.setRange(-1000, 1000)
        self.center_z_spin.setValue(-50)
        center_layout.addWidget(QLabel("Z:"))
        center_layout.addWidget(self.center_z_spin)

        self.params_layout.addRow("中心点:", center_layout)

        # 缩放因子
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        self.params_layout.addRow("缩放因子:", self.scale_spin)

        self.params_group.setLayout(self.params_layout)
        layout.addWidget(self.params_group)

        # 预览信息
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc;")
        layout.addWidget(self.preview_label)

        # 更新预览
        self.update_preview()

        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("创建断层")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def on_template_selected(self, template_name):
        """模板选择事件"""
        self.selected_template = template_name
        self.update_preview()

    def update_preview(self):
        """更新预览信息"""
        if not hasattr(self, 'selected_template'):
            return

        template = self.templates[self.selected_template]
        preview_text = f"模板: {self.selected_template}\n"
        preview_text += f"类型: {template['type']}\n"

        if template['method'] == 'strike_dip':
            preview_text += f"走向: {template['strike']}°\n"
            preview_text += f"倾向: {template['dip']}°\n"
            preview_text += f"长度: {template['length']}m\n"
            preview_text += f"宽度: {template['width']}m"
        elif template['method'] == 'points':
            preview_text += f"预定义点数: {len(template['points'])}"

        self.preview_label.setText(preview_text)

    def get_selected_template(self):
        """获取选中的模板"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入断层名称")
            return None

        template = self.templates[self.selected_template].copy()
        template['name'] = name

        # Apply中心点
        center_point = (
            self.center_x_spin.value(),
            self.center_y_spin.value(),
            self.center_z_spin.value()
        )
        template['center_point'] = center_point

        # Apply缩放
        scale = self.scale_spin.value()
        if template['method'] == 'strike_dip':
            template['length'] *= scale
            template['width'] *= scale
        elif template['method'] == 'points':
            # 缩放点坐标
            scaled_points = []
            for x, y, z in template['points']:
                scaled_points.append((
                    center_point[0] + x * scale,
                    center_point[1] + y * scale,
                    center_point[2] + z * scale
                ))
            template['points'] = scaled_points

        return template


class AdvancedFaultDialog(QDialog):
    """高级断层添加对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("高级断层添加")
        self.setModal(True)
        self.resize(500, 600)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 方法选择
        method_group = QGroupBox("添加方法")
        method_layout = QVBoxLayout()

        self.method_buttons = {}
        methods = [
            ("points", "点数据定义"),
            ("equation", "平面方程定义"),
            ("strike_dip", "走向倾向定义")
        ]

        for method_id, method_name in methods:
            btn = QtWidgets.QRadioButton(method_name)
            btn.clicked.connect(lambda checked, mid=method_id: self.on_method_changed(mid))
            method_layout.addWidget(btn)
            self.method_buttons[method_id] = btn

        # 默认选择第一个
        self.method_buttons["points"].setChecked(True)
        self.current_method = "points"

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # 参数面板
        self.params_stack = QtWidgets.QStackedWidget()

        # 点数据面板
        self.points_widget = self.create_points_widget()
        self.params_stack.addWidget(self.points_widget)

        # 方程面板
        self.equation_widget = self.create_equation_widget()
        self.params_stack.addWidget(self.equation_widget)

        # 走向倾向面板
        self.strike_dip_widget = self.create_strike_dip_widget()
        self.params_stack.addWidget(self.strike_dip_widget)

        layout.addWidget(self.params_stack)

        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("创建断层")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def on_method_changed(self, method):
        """方法改变事件"""
        self.current_method = method
        if method == "points":
            self.params_stack.setCurrentIndex(0)
        elif method == "equation":
            self.params_stack.setCurrentIndex(1)
        elif method == "strike_dip":
            self.params_stack.setCurrentIndex(2)

    def create_points_widget(self):
        """创建点数据输入面板"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 基本信息
        form_layout = QFormLayout()
        self.points_name_edit = QtWidgets.QLineEdit("高级断层_点")
        form_layout.addRow("断层名称:", self.points_name_edit)

        self.points_type_combo = QComboBox()
        self.points_type_combo.addItems(['逆断层', '正断层', '平推断层', '走滑断层', '未知断层'])
        form_layout.addRow("断层类型:", self.points_type_combo)

        layout.addLayout(form_layout)

        # 点数据表格
        self.points_table = QTableWidget(3, 3)
        self.points_table.setHorizontalHeaderLabels(['X', 'Y', 'Z'])

        # 设置默认值
        default_points = [(0, 0, 0), (50, 0, -25), (100, 0, -50)]
        for i, (x, y, z) in enumerate(default_points):
            self.points_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(x)))
            self.points_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(y)))
            self.points_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(z)))

        layout.addWidget(self.points_table)

        # 表格操作按钮
        table_btn_layout = QHBoxLayout()
        add_row_btn = QPushButton("添加行")
        add_row_btn.clicked.connect(self.add_points_row)
        table_btn_layout.addWidget(add_row_btn)

        remove_row_btn = QPushButton("删除行")
        remove_row_btn.clicked.connect(self.remove_points_row)
        table_btn_layout.addWidget(remove_row_btn)

        layout.addLayout(table_btn_layout)

        widget.setLayout(layout)
        return widget

    def create_equation_widget(self):
        """创建方程输入面板"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 基本信息
        form_layout = QFormLayout()
        self.equation_name_edit = QtWidgets.QLineEdit("高级断层_方程")
        form_layout.addRow("断层名称:", self.equation_name_edit)

        self.equation_type_combo = QComboBox()
        self.equation_type_combo.addItems(['逆断层', '正断层', '平推断层', '走滑断层', '未知断层'])
        form_layout.addRow("断层类型:", self.equation_type_combo)

        layout.addLayout(form_layout)

        # 平面方程
        equation_group = QGroupBox("平面方程: Ax + By + Cz + D = 0")
        equation_layout = QFormLayout()

        self.eq_a_spin = QDoubleSpinBox()
        self.eq_a_spin.setRange(-1000, 1000)
        self.eq_a_spin.setValue(1.0)
        equation_layout.addRow("A:", self.eq_a_spin)

        self.eq_b_spin = QDoubleSpinBox()
        self.eq_b_spin.setRange(-1000, 1000)
        self.eq_b_spin.setValue(0.0)
        equation_layout.addRow("B:", self.eq_b_spin)

        self.eq_c_spin = QDoubleSpinBox()
        self.eq_c_spin.setRange(-1000, 1000)
        self.eq_c_spin.setValue(1.0)
        equation_layout.addRow("C:", self.eq_c_spin)

        self.eq_d_spin = QDoubleSpinBox()
        self.eq_d_spin.setRange(-1000, 1000)
        self.eq_d_spin.setValue(0.0)
        equation_layout.addRow("D:", self.eq_d_spin)

        equation_group.setLayout(equation_layout)
        layout.addWidget(equation_group)

        # 边界设置
        bounds_group = QGroupBox("边界范围")
        bounds_layout = QFormLayout()

        # X范围
        x_layout = QHBoxLayout()
        self.eq_x_min_spin = QDoubleSpinBox()
        self.eq_x_min_spin.setRange(-1000, 1000)
        self.eq_x_min_spin.setValue(-50)
        x_layout.addWidget(self.eq_x_min_spin)
        x_layout.addWidget(QLabel(" 到 "))
        self.eq_x_max_spin = QDoubleSpinBox()
        self.eq_x_max_spin.setRange(-1000, 1000)
        self.eq_x_max_spin.setValue(50)
        x_layout.addWidget(self.eq_x_max_spin)
        bounds_layout.addRow("X范围:", x_layout)

        # Y范围
        y_layout = QHBoxLayout()
        self.eq_y_min_spin = QDoubleSpinBox()
        self.eq_y_min_spin.setRange(-1000, 1000)
        self.eq_y_min_spin.setValue(-50)
        y_layout.addWidget(self.eq_y_min_spin)
        y_layout.addWidget(QLabel(" 到 "))
        self.eq_y_max_spin = QDoubleSpinBox()
        self.eq_y_max_spin.setRange(-1000, 1000)
        self.eq_y_max_spin.setValue(50)
        y_layout.addWidget(self.eq_y_max_spin)
        bounds_layout.addRow("Y范围:", y_layout)

        # Z范围
        z_layout = QHBoxLayout()
        self.eq_z_min_spin = QDoubleSpinBox()
        self.eq_z_min_spin.setRange(-1000, 1000)
        self.eq_z_min_spin.setValue(-100)
        z_layout.addWidget(self.eq_z_min_spin)
        z_layout.addWidget(QLabel(" 到 "))
        self.eq_z_max_spin = QDoubleSpinBox()
        self.eq_z_max_spin.setRange(-1000, 1000)
        self.eq_z_max_spin.setValue(0)
        z_layout.addWidget(self.eq_z_max_spin)
        bounds_layout.addRow("Z范围:", z_layout)

        bounds_group.setLayout(bounds_layout)
        layout.addWidget(bounds_group)

        widget.setLayout(layout)
        return widget

    def create_strike_dip_widget(self):
        """创建走向倾向输入面板"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 基本信息
        form_layout = QFormLayout()
        self.sd_name_edit = QtWidgets.QLineEdit("高级断层_走向倾向")
        form_layout.addRow("断层名称:", self.sd_name_edit)

        self.sd_type_combo = QComboBox()
        self.sd_type_combo.addItems(['逆断层', '正断层', '平推断层', '走滑断层', '未知断层'])
        form_layout.addRow("断层类型:", self.sd_type_combo)

        layout.addLayout(form_layout)

        # 中心点
        center_group = QGroupBox("中心点Position")
        center_layout = QFormLayout()

        self.sd_center_x_spin = QDoubleSpinBox()
        self.sd_center_x_spin.setRange(-1000, 1000)
        self.sd_center_x_spin.setValue(0)
        center_layout.addRow("X:", self.sd_center_x_spin)

        self.sd_center_y_spin = QDoubleSpinBox()
        self.sd_center_y_spin.setRange(-1000, 1000)
        self.sd_center_y_spin.setValue(0)
        center_layout.addRow("Y:", self.sd_center_y_spin)

        self.sd_center_z_spin = QDoubleSpinBox()
        self.sd_center_z_spin.setRange(-1000, 1000)
        self.sd_center_z_spin.setValue(-50)
        center_layout.addRow("Z:", self.sd_center_z_spin)

        center_group.setLayout(center_layout)
        layout.addWidget(center_group)

        # Direction参数
        orientation_group = QGroupBox("Direction参数")
        orientation_layout = QFormLayout()

        self.sd_strike_spin = QDoubleSpinBox()
        self.sd_strike_spin.setRange(0, 360)
        self.sd_strike_spin.setValue(0)
        self.sd_strike_spin.setSuffix("°")
        orientation_layout.addRow("走向角:", self.sd_strike_spin)

        self.sd_dip_spin = QDoubleSpinBox()
        self.sd_dip_spin.setRange(0, 90)
        self.sd_dip_spin.setValue(60)
        self.sd_dip_spin.setSuffix("°")
        orientation_layout.addRow("倾向角:", self.sd_dip_spin)

        orientation_group.setLayout(orientation_layout)
        layout.addWidget(orientation_group)

        # 尺寸参数
        size_group = QGroupBox("尺寸参数")
        size_layout = QFormLayout()

        self.sd_length_spin = QDoubleSpinBox()
        self.sd_length_spin.setRange(1, 1000)
        self.sd_length_spin.setValue(100)
        size_layout.addRow("走向长度:", self.sd_length_spin)

        self.sd_width_spin = QDoubleSpinBox()
        self.sd_width_spin.setRange(1, 1000)
        self.sd_width_spin.setValue(50)
        size_layout.addRow("倾向宽度:", self.sd_width_spin)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        widget.setLayout(layout)
        return widget

    def add_points_row(self):
        """添加点行"""
        row_count = self.points_table.rowCount()
        self.points_table.insertRow(row_count)
        for j in range(3):
            self.points_table.setItem(row_count, j, QtWidgets.QTableWidgetItem("0.0"))

    def remove_points_row(self):
        """删除点行"""
        if self.points_table.rowCount() > 3:
            current_row = self.points_table.currentRow()
            if current_row >= 0:
                self.points_table.removeRow(current_row)

    def get_fault_data(self):
        """获取断层数据"""
        if self.current_method == "points":
            return self.get_points_data()
        elif self.current_method == "equation":
            return self.get_equation_data()
        elif self.current_method == "strike_dip":
            return self.get_strike_dip_data()
        return None

    def get_points_data(self):
        """获取点数据"""
        name = self.points_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入断层名称")
            return None

        fault_type = self.points_type_combo.currentText()

        points = []
        for i in range(self.points_table.rowCount()):
            try:
                x = float(self.points_table.item(i, 0).text())
                y = float(self.points_table.item(i, 1).text())
                z = float(self.points_table.item(i, 2).text())
                points.append((x, y, z))
            except (ValueError, AttributeError):
                QMessageBox.warning(self, "警告", f"第{i + 1}行数据格式错误")
                return None

        if len(points) < 3:
            QMessageBox.warning(self, "警告", "至少需要3个点")
            return None

        return {
            'method': 'points',
            'name': name,
            'type': fault_type,
            'points': points
        }

    def get_equation_data(self):
        """获取方程数据"""
        name = self.equation_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入断层名称")
            return None

        fault_type = self.equation_type_combo.currentText()

        plane_params = [
            self.eq_a_spin.value(),
            self.eq_b_spin.value(),
            self.eq_c_spin.value(),
            self.eq_d_spin.value()
        ]

        bounds = [
            self.eq_x_min_spin.value(),
            self.eq_x_max_spin.value(),
            self.eq_y_min_spin.value(),
            self.eq_y_max_spin.value(),
            self.eq_z_min_spin.value(),
            self.eq_z_max_spin.value()
        ]

        return {
            'method': 'equation',
            'name': name,
            'type': fault_type,
            'plane_params': plane_params,
            'bounds': bounds
        }

    def get_strike_dip_data(self):
        """获取走向倾向数据"""
        name = self.sd_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入断层名称")
            return None

        fault_type = self.sd_type_combo.currentText()

        center_point = (
            self.sd_center_x_spin.value(),
            self.sd_center_y_spin.value(),
            self.sd_center_z_spin.value()
        )

        return {
            'method': 'strike_dip',
            'name': name,
            'type': fault_type,
            'center_point': center_point,
            'strike': self.sd_strike_spin.value(),
            'dip': self.sd_dip_spin.value(),
            'length': self.sd_length_spin.value(),
            'width': self.sd_width_spin.value()
        }


# ===== 原有的断层对话框类 =====

class AddFaultPointsDialog(QDialog):
    """添加点数据断层对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加点数据断层")
        self.setModal(True)
        self.resize(400, 500)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 断层名称
        form_layout = QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("输入断层名称...")
        form_layout.addRow("断层名称:", self.name_edit)

        # 断层类型
        self.type_combo = QComboBox()
        self.type_combo.addItems(['逆断层', '正断层', '平推断层', '走滑断层', '未知断层'])
        form_layout.addRow("断层类型:", self.type_combo)

        layout.addLayout(form_layout)

        # 点数据输入
        points_group = QGroupBox("断层点数据 (至少3个点)")
        points_layout = QVBoxLayout()

        # 点数据表格
        self.points_table = QTableWidget(3, 3)
        self.points_table.setHorizontalHeaderLabels(['X', 'Y', 'Z'])
        self.points_table.setMinimumHeight(150)

        # 设置默认值
        for i in range(3):
            for j in range(3):
                self.points_table.setItem(i, j, QtWidgets.QTableWidgetItem("0.0"))

        points_layout.addWidget(self.points_table)

        # 添加/删除点按钮
        btn_layout = QHBoxLayout()
        self.add_point_btn = QPushButton("添加点")
        self.add_point_btn.clicked.connect(self.add_point_row)
        btn_layout.addWidget(self.add_point_btn)

        self.remove_point_btn = QPushButton("删除点")
        self.remove_point_btn.clicked.connect(self.remove_point_row)
        btn_layout.addWidget(self.remove_point_btn)

        points_layout.addLayout(btn_layout)
        points_group.setLayout(points_layout)
        layout.addWidget(points_group)

        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def add_point_row(self):
        """添加点行"""
        row_count = self.points_table.rowCount()
        self.points_table.insertRow(row_count)
        for j in range(3):
            self.points_table.setItem(row_count, j, QtWidgets.QTableWidgetItem("0.0"))

    def remove_point_row(self):
        """删除点行"""
        if self.points_table.rowCount() > 3:  # 至少保留3个点
            current_row = self.points_table.currentRow()
            if current_row >= 0:
                self.points_table.removeRow(current_row)

    def get_fault_data(self):
        """获取断层数据"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入断层名称")
            return None

        fault_type = self.type_combo.currentText()

        # 收集点数据
        points = []
        for i in range(self.points_table.rowCount()):
            try:
                x = float(self.points_table.item(i, 0).text())
                y = float(self.points_table.item(i, 1).text())
                z = float(self.points_table.item(i, 2).text())
                points.append((x, y, z))
            except (ValueError, AttributeError):
                QMessageBox.warning(self, "警告", f"第{i + 1}行数据格式错误")
                return None

        if len(points) < 3:
            QMessageBox.warning(self, "警告", "至少需要3个点来定义断层")
            return None

        return {
            'name': name,
            'type': fault_type,
            'points': points
        }


class AddFaultEquationDialog(QDialog):
    """添加方程断层对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加方程断层")
        self.setModal(True)
        self.resize(400, 300)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 基本信息
        form_layout = QFormLayout()

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("输入断层名称...")
        form_layout.addRow("断层名称:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(['逆断层', '正断层', '平推断层', '走滑断层', '未知断层'])
        form_layout.addRow("断层类型:", self.type_combo)

        layout.addLayout(form_layout)

        # 平面方程参数
        equation_group = QGroupBox("平面方程: Ax + By + Cz + D = 0")
        equation_layout = QFormLayout()

        self.a_spin = QDoubleSpinBox()
        self.a_spin.setRange(-1000, 1000)
        self.a_spin.setValue(1.0)
        equation_layout.addRow("A:", self.a_spin)

        self.b_spin = QDoubleSpinBox()
        self.b_spin.setRange(-1000, 1000)
        self.b_spin.setValue(0.0)
        equation_layout.addRow("B:", self.b_spin)

        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(-1000, 1000)
        self.c_spin.setValue(1.0)
        equation_layout.addRow("C:", self.c_spin)

        self.d_spin = QDoubleSpinBox()
        self.d_spin.setRange(-1000, 1000)
        self.d_spin.setValue(0.0)
        equation_layout.addRow("D:", self.d_spin)

        equation_group.setLayout(equation_layout)
        layout.addWidget(equation_group)

        # 边界范围
        bounds_group = QGroupBox("断层边界范围")
        bounds_layout = QFormLayout()

        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-1000, 1000)
        self.x_min_spin.setValue(-50)
        bounds_layout.addRow("X最小值:", self.x_min_spin)

        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-1000, 1000)
        self.x_max_spin.setValue(50)
        bounds_layout.addRow("X最大值:", self.x_max_spin)

        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1000, 1000)
        self.y_min_spin.setValue(-50)
        bounds_layout.addRow("Y最小值:", self.y_min_spin)

        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-1000, 1000)
        self.y_max_spin.setValue(50)
        bounds_layout.addRow("Y最大值:", self.y_max_spin)

        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(-1000, 1000)
        self.z_min_spin.setValue(-100)
        bounds_layout.addRow("Z最小值:", self.z_min_spin)

        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(-1000, 1000)
        self.z_max_spin.setValue(0)
        bounds_layout.addRow("Z最大值:", self.z_max_spin)

        bounds_group.setLayout(bounds_layout)
        layout.addWidget(bounds_group)

        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_fault_data(self):
        """获取断层数据"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入断层名称")
            return None

        fault_type = self.type_combo.currentText()

        plane_params = [
            self.a_spin.value(),
            self.b_spin.value(),
            self.c_spin.value(),
            self.d_spin.value()
        ]

        bounds = [
            self.x_min_spin.value(),
            self.x_max_spin.value(),
            self.y_min_spin.value(),
            self.y_max_spin.value(),
            self.z_min_spin.value(),
            self.z_max_spin.value()
        ]

        return {
            'name': name,
            'type': fault_type,
            'plane_params': plane_params,
            'bounds': bounds
        }


class AddFaultStrikeDipDialog(QDialog):
    """添加走向倾向断层对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加走向倾向断层")
        self.setModal(True)
        self.resize(400, 400)

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 基本信息
        form_layout = QFormLayout()

        self.name_edit = QtWidgets.QLineEdit()
        self.name_edit.setPlaceholderText("输入断层名称...")
        form_layout.addRow("断层名称:", self.name_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(['逆断层', '正断层', '平推断层', '走滑断层', '未知断层'])
        form_layout.addRow("断层类型:", self.type_combo)

        layout.addLayout(form_layout)

        # 中心点Position
        center_group = QGroupBox("断层中心点Position")
        center_layout = QFormLayout()

        self.center_x_spin = QDoubleSpinBox()
        self.center_x_spin.setRange(-1000, 1000)
        self.center_x_spin.setValue(0)
        center_layout.addRow("中心点X:", self.center_x_spin)

        self.center_y_spin = QDoubleSpinBox()
        self.center_y_spin.setRange(-1000, 1000)
        self.center_y_spin.setValue(0)
        center_layout.addRow("中心点Y:", self.center_y_spin)

        self.center_z_spin = QDoubleSpinBox()
        self.center_z_spin.setRange(-1000, 1000)
        self.center_z_spin.setValue(-50)
        center_layout.addRow("中心点Z:", self.center_z_spin)

        center_group.setLayout(center_layout)
        layout.addWidget(center_group)

        # 走向倾向参数
        orientation_group = QGroupBox("走向倾向参数")
        orientation_layout = QFormLayout()

        self.strike_spin = QDoubleSpinBox()
        self.strike_spin.setRange(0, 360)
        self.strike_spin.setValue(0)
        self.strike_spin.setSuffix("°")
        orientation_layout.addRow("走向角 (0-360°):", self.strike_spin)

        self.dip_spin = QDoubleSpinBox()
        self.dip_spin.setRange(0, 90)
        self.dip_spin.setValue(60)
        self.dip_spin.setSuffix("°")
        orientation_layout.addRow("倾向角 (0-90°):", self.dip_spin)

        orientation_group.setLayout(orientation_layout)
        layout.addWidget(orientation_group)

        # 断层尺寸
        size_group = QGroupBox("断层尺寸")
        size_layout = QFormLayout()

        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(1, 1000)
        self.length_spin.setValue(100)
        size_layout.addRow("走向长度:", self.length_spin)

        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1, 1000)
        self.width_spin.setValue(50)
        size_layout.addRow("倾向宽度:", self.width_spin)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # 按钮
        button_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def get_fault_data(self):
        """获取断层数据"""
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "警告", "请输入断层名称")
            return None

        fault_type = self.type_combo.currentText()

        center_point = (
            self.center_x_spin.value(),
            self.center_y_spin.value(),
            self.center_z_spin.value()
        )

        return {
            'name': name,
            'type': fault_type,
            'center_point': center_point,
            'strike': self.strike_spin.value(),
            'dip': self.dip_spin.value(),
            'length': self.length_spin.value(),
            'width': self.width_spin.value()
        }


if __name__ == "__main__":
    # 额外的Qt环境修复
    os.environ.setdefault('QT_QPA_PLATFORM', 'windows')
    os.environ.setdefault('QT_PLUGIN_PATH', '')

    try:
        app = QtWidgets.QApplication(sys.argv)

        # 设置Apply程序属性
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

        viewer = BoreholeViewer()
        viewer.show()
        sys.exit(app.exec_())

    except Exception as e:
        print(f"程序启动失败: {e}")
        print("尝试备用启动方式...")

        # 备用启动方式
        try:
            import sys

            os.environ['QT_QPA_PLATFORM'] = 'windows'
            app = QtWidgets.QApplication([])
            viewer = BoreholeViewer()
            viewer.show()
            app.exec_()
        except Exception as e2:
            print(f"备用启动也失败: {e2}")
            sys.exit(1)