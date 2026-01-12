# 1. 正确导入：区分 FastAPI 的 Path 和 pathlib 的 Path
from fastapi import FastAPI, HTTPException, Path  # 导入 FastAPI 的路径参数工具
from fastapi.responses import FileResponse
import subprocess
import shutil
from pathlib import Path as PathlibPath  # 重命名 pathlib 的 Path，避免混淆
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import logging
from urllib.parse import unquote
import pyarrow.parquet as pq
import glob
from urllib.parse import quote

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 文件路径定义：用重命名后的 PathlibPath（pathlib 的功能）
INDEX_ROOT = PathlibPath("/app/indexing")
INPUT_DIR = INDEX_ROOT / "input"  # ground 目录
COPYPATH_ROOT = PathlibPath("/app/saves/data/copypath")
DRILL_INDEX_ROOT = PathlibPath("/app/indexing_drill")
DRILL_INPUT_DIR = DRILL_INDEX_ROOT / "input"  # drill 目录
# 定义可下载文件的目录
GROUND_DOWNLOAD_DIR = INDEX_ROOT / "ground_graph_fill"
DRILL_DOWNLOAD_DIR = DRILL_INDEX_ROOT / "drill_graph_fill"
DIAMAGNETIC_INPUT_DIR = PathlibPath("/app/indexing_diamagnetic/input")


def convert_to_graph_format(input_csv_path: PathlibPath, output_csv_path: PathlibPath) -> dict:
    """
    将知识图谱CSV文件转换为图数据库上传格式

    Args:
        input_csv_path: 输入的CSV文件路径
        output_csv_path: 输出的CSV文件路径

    Returns:
        dict: 转换结果信息
    """
    try:
        # 检查输入文件是否存在
        if not input_csv_path.exists():
            return {"status": "error", "detail": f"输入文件不存在: {input_csv_path}"}

        # 读取CSV文件
        print(f"📖 正在读取文件: {input_csv_path}")
        df = pd.read_csv(input_csv_path)

        # 检查必需的列是否存在
        required_columns = ['source', 'target', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            return {
                "status": "error",
                "detail": f"CSV文件中缺少必需的列: {missing_columns}",
                "available_columns": list(df.columns)
            }

        # 提取需要的列并重命名
        graph_df = df[['source', 'description', 'target']].copy()
        graph_df.columns = ['h', 'r', 't']  # 重命名为图数据库要求的格式

        # 确保输出目录存在
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为新的CSV文件
        graph_df.to_csv(output_csv_path, index=False, encoding='utf-8')

        print(f"✅ 已成功转换文件格式")
        print(f"📊 转换统计: {len(graph_df)} 条关系")
        print(f"💾 输出文件: {output_csv_path}")

        return {
            "status": "success",
            "detail": "文件格式转换成功",
            "input_file": str(input_csv_path),
            "output_file": str(output_csv_path),
            "relationship_count": len(graph_df),
            "sample_data": graph_df.head(3).to_dict('records')  # 返回前3条数据作为示例
        }

    except Exception as e:
        return {"status": "error", "detail": f"文件格式转换失败: {str(e)}"}

def convert_parquet_to_csv(parquet_path: PathlibPath, csv_path: PathlibPath) -> dict:
    """
    将 Parquet 文件转换为 CSV 文件

    Args:
        parquet_path: Parquet 文件路径
        csv_path: 输出的 CSV 文件路径

    Returns:
        dict: 转换结果信息
    """
    try:
        # 检查输入文件是否存在
        if not parquet_path.exists():
            return {"status": "error", "detail": f"Parquet 文件不存在: {parquet_path}"}

        # 确保输出目录存在
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 读取 Parquet 文件
        print(f"📖 正在读取 Parquet 文件: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        # 转换为 CSV
        print(f"💾 正在写入 CSV 文件: {csv_path}")
        df.to_csv(csv_path, index=False, encoding='utf-8')

        # 返回转换信息
        return {
            "status": "success",
            "detail": "文件转换成功",
            "original_file": str(parquet_path),
            "converted_file": str(csv_path),
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        }

    except Exception as e:
        return {"status": "error", "detail": f"文件转换失败: {str(e)}"}


def find_latest_output_dir(base_dir: PathlibPath) -> PathlibPath:
    """
    在 output 目录中找到最新的时间戳目录

    Args:
        base_dir: 基础目录（INDEX_ROOT 或 DRILL_INDEX_ROOT）

    Returns:
        PathlibPath: 最新的时间戳目录路径
    """
    output_dir = base_dir / "output"
    if not output_dir.exists():
        return None

    # 获取所有时间戳目录并排序（最新的在前）
    timestamp_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()], reverse=True)

    if not timestamp_dirs:
        return None

    return timestamp_dirs[0]
# ------------------- 原有接口（无路径参数，无需修改）-------------------
@app.post("/init_index")
def init_index():
    """创建索引（保持不变）"""
    try:
        subprocess.run(
            ["python", "-m", "graphrag.index", "--init", "--root", str(INDEX_ROOT)],
            check=True
        )
        return {"status": "索引创建成功"}
    except subprocess.CalledProcessError as e:
        return {"status": "索引创建失败", "detail": str(e)}


@app.post("/build_graph")
def build_graph(clean_copypath: bool = True):
    """构建 ground 图谱"""
    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ 确保 input 目录存在: {INPUT_DIR}")

        if not COPYPATH_ROOT.exists():
            raise HTTPException(status_code=400, detail=f"copypath 目录不存在: {COPYPATH_ROOT}")

        copypath_files = [f for f in COPYPATH_ROOT.iterdir() if f.is_file()]
        if not copypath_files:
            raise HTTPException(status_code=400, detail=f"copypath 目录中无文件可复制: {COPYPATH_ROOT}")

        for file in copypath_files:
            target_file = INPUT_DIR / file.name
            shutil.copy2(file, target_file)
            print(f"📁 已复制文件: {file.name} → {INPUT_DIR}")

        print("✅ 图谱构建即将执行")
        subprocess.run(
            ["python", "-m", "graphrag.index", "--root", str(INDEX_ROOT)],
            check=True,
        )
        print("✅ 图谱构建命令执行成功")

        # 新增：查找并转换 Parquet 文件
        conversion_result = None
        graph_conversion_result = None

        # 确保下载目录存在
        GROUND_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 查找最新的输出目录
        latest_output_dir = find_latest_output_dir(INDEX_ROOT)

        if latest_output_dir:
            parquet_file = latest_output_dir / "artifacts" / "create_final_relationships.parquet"
            if parquet_file.exists():
                # 生成 CSV 文件名
                csv_filename = "create_final_relationships.csv"
                csv_file = GROUND_DOWNLOAD_DIR / csv_filename

                # 转换文件
                conversion_result = convert_parquet_to_csv(parquet_file, csv_file)
                print(f"📊 Ground 图谱转换结果: {conversion_result['status']}")

                # 如果转换成功，打印详细信息并进行图数据库格式转换
                if conversion_result["status"] == "success":
                    print(f"✅ 已转换: {parquet_file} → {csv_file}")
                    print(f"📊 数据统计: {conversion_result['rows']} 行, {conversion_result['columns']} 列")

                    # 直接进行图数据库格式转换
                    graph_csv_filename = "graph_format_relationships.csv"
                    graph_csv_file = GROUND_DOWNLOAD_DIR / graph_csv_filename

                    graph_conversion_result = convert_to_graph_format(csv_file, graph_csv_file)
                    print(f"📊 图数据库格式转换结果: {graph_conversion_result['status']}")

                    # 如果图数据库格式转换成功，删除原始的CSV文件，只保留规范化的文件
                    if graph_conversion_result["status"] == "success":
                        # 删除原始的CSV文件
                        csv_file.unlink()
                        print(f"🗑️ 已删除原始CSV文件: {csv_file}")

                        # 将规范化文件重命名为原始文件名
                        final_csv_file = GROUND_DOWNLOAD_DIR / csv_filename
                        graph_csv_file.rename(final_csv_file)
                        print(f"📝 已将规范化文件重命名为: {final_csv_file}")

                        # 更新转换结果信息
                        conversion_result["converted_file"] = str(final_csv_file)
                        conversion_result["detail"] = "文件已转换为图数据库格式并保存"
            else:
                conversion_result = {"status": "warning", "detail": f"未找到 Parquet 文件: {parquet_file}"}
                print(f"⚠️ {conversion_result['detail']}")

                # 列出 artifacts 目录中的所有文件，方便调试
                artifacts_dir = latest_output_dir / "artifacts"
                if artifacts_dir.exists():
                    artifact_files = [f.name for f in artifacts_dir.iterdir() if f.is_file()]
                    print(f"📁 artifacts 目录中的文件: {artifact_files}")
        else:
            conversion_result = {"status": "warning", "detail": "未找到最新的 output 目录"}
            print(f"⚠️ {conversion_result['detail']}")

            # 列出 output 目录中的所有子目录，方便调试
            output_dir = INDEX_ROOT / "output"
            if output_dir.exists():
                subdirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
                print(f"📁 output 目录中的子目录: {subdirs}")

        if clean_copypath:
            for file in copypath_files:
                file.unlink()
            print(f"✅ 已清空 copypath 目录: {COPYPATH_ROOT}")

        response_data = {
            "status": "图谱构建成功",
            "detail": f"已复制 {len(copypath_files)} 个文件到 indexing/input"
        }

        if conversion_result:
            response_data["csv_conversion"] = conversion_result

        if graph_conversion_result:
            response_data["graph_format_conversion"] = graph_conversion_result

        return response_data

    except subprocess.CalledProcessError as e:
        return {
            "status": "图谱构建失败",
            "detail": f"命令执行错误: {str(e)}",
            "错误输出": e.stderr.decode('utf-8') if e.stderr else "无详细错误信息"
        }
    except Exception as e:
        return {"status": "操作失败", "detail": str(e)}


@app.post("/build_drillgraph")
def build_drillgraph(clean_copypath: bool = True):
    """构建 drill 图谱"""
    try:
        DRILL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ 确保 drill input 目录存在: {DRILL_INPUT_DIR}")

        if not COPYPATH_ROOT.exists():
            raise HTTPException(status_code=400, detail=f"copypath 目录不存在: {COPYPATH_ROOT}")

        copypath_files = [f for f in COPYPATH_ROOT.iterdir() if f.is_file()]
        if not copypath_files:
            raise HTTPException(status_code=400, detail=f"copypath 目录中无文件可复制: {COPYPATH_ROOT}")

        for file in copypath_files:
            target_file = DRILL_INPUT_DIR / file.name
            shutil.copy2(file, target_file)
            print(f"📁 已复制文件: {file.name} → {DRILL_INPUT_DIR}")

        print("✅ Drill 图谱构建即将执行")
        subprocess.run(
            ["python", "-m", "graphrag.index", "--root", str(DRILL_INDEX_ROOT)],
            check=True,
        )
        print("✅ Drill 图谱构建命令执行成功")

        # 新增：查找并转换 Parquet 文件
        conversion_result = None
        graph_conversion_result = None

        # 确保下载目录存在
        DRILL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 查找最新的输出目录
        latest_output_dir = find_latest_output_dir(DRILL_INDEX_ROOT)

        if latest_output_dir:
            parquet_file = latest_output_dir / "artifacts" / "create_final_relationships.parquet"
            if parquet_file.exists():
                # 生成 CSV 文件名
                csv_filename = "create_final_relationships.csv"
                csv_file = DRILL_DOWNLOAD_DIR / csv_filename

                # 转换文件
                conversion_result = convert_parquet_to_csv(parquet_file, csv_file)
                print(f"📊 Drill 图谱转换结果: {conversion_result['status']}")

                # 如果转换成功，打印详细信息并进行图数据库格式转换
                if conversion_result["status"] == "success":
                    print(f"✅ 已转换: {parquet_file} → {csv_file}")
                    print(f"📊 数据统计: {conversion_result['rows']} 行, {conversion_result['columns']} 列")

                    # 直接进行图数据库格式转换
                    graph_csv_filename = "graph_format_relationships.csv"
                    graph_csv_file = DRILL_DOWNLOAD_DIR / graph_csv_filename

                    graph_conversion_result = convert_to_graph_format(csv_file, graph_csv_file)
                    print(f"📊 图数据库格式转换结果: {graph_conversion_result['status']}")

                    # 如果图数据库格式转换成功，删除原始的CSV文件，只保留规范化的文件
                    if graph_conversion_result["status"] == "success":
                        # 删除原始的CSV文件
                        csv_file.unlink()
                        print(f"🗑️ 已删除原始CSV文件: {csv_file}")

                        # 将规范化文件重命名为原始文件名
                        final_csv_file = DRILL_DOWNLOAD_DIR / csv_filename
                        graph_csv_file.rename(final_csv_file)
                        print(f"📝 已将规范化文件重命名为: {final_csv_file}")

                        # 更新转换结果信息
                        conversion_result["converted_file"] = str(final_csv_file)
                        conversion_result["detail"] = "文件已转换为图数据库格式并保存"
            else:
                conversion_result = {"status": "warning", "detail": f"未找到 Parquet 文件: {parquet_file}"}
                print(f"⚠️ {conversion_result['detail']}")

                # 列出 artifacts 目录中的所有文件，方便调试
                artifacts_dir = latest_output_dir / "artifacts"
                if artifacts_dir.exists():
                    artifact_files = [f.name for f in artifacts_dir.iterdir() if f.is_file()]
                    print(f"📁 artifacts 目录中的文件: {artifact_files}")
        else:
            conversion_result = {"status": "warning", "detail": "未找到最新的 output 目录"}
            print(f"⚠️ {conversion_result['detail']}")

            # 列出 output 目录中的所有子目录，方便调试
            output_dir = DRILL_INDEX_ROOT / "output"
            if output_dir.exists():
                subdirs = [d.name for d in output_dir.iterdir() if d.is_dir()]
                print(f"📁 output 目录中的子目录: {subdirs}")

        if clean_copypath:
            for file in copypath_files:
                file.unlink()
            print(f"✅ 已清空 copypath 目录: {COPYPATH_ROOT}")

        response_data = {
            "status": "Drill 图谱构建成功",
            "detail": f"已复制 {len(copypath_files)} 个文件到 indexing_drill/input"
        }

        if conversion_result:
            response_data["csv_conversion"] = conversion_result

        if graph_conversion_result:
            response_data["graph_format_conversion"] = graph_conversion_result

        return response_data

    except subprocess.CalledProcessError as e:
        return {
            "status": "Drill 图谱构建失败",
            "detail": f"命令执行错误: {str(e)}",
            "错误输出": e.stderr.decode('utf-8') if e.stderr else "无详细错误信息"
        }
    except Exception as e:
        return {"status": "操作失败", "detail": str(e)}


# ------------------- 修正路径参数的接口 -------------------
@app.get("/get_file_list/{directory_type}")
def get_file_list(
        # 用 FastAPI 的 Path 定义路径参数（必填、描述、正则验证）
        directory_type: str = Path(..., description="要查询的目录类型", regex="^(drill|ground)$")
):
    """获取指定目录（drill/ground）下的文件列表"""
    try:
        # 用 PathlibPath 处理文件路径
        if directory_type == "drill":
            target_dir = DRILL_INPUT_DIR
        else:  # ground
            target_dir = INPUT_DIR

        if not target_dir.exists() or not target_dir.is_dir():
            raise HTTPException(status_code=404, detail=f"{directory_type} 目录不存在: {target_dir}")

        file_list = []
        for file_path in target_dir.iterdir():
            if file_path.is_file():
                file_list.append({
                    "file_name": file_path.name,
                    "size_bytes": file_path.stat().st_size
                })

        return {
            "status": "success",
            "directory": str(target_dir),
            "file_count": len(file_list),
            "files": file_list
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "detail": f"获取文件列表失败: {str(e)}"}

@app.delete("/delete_file/{directory_type}/{file_name}")
def delete_specific_file(
        directory_type: str = Path(..., description="文件所在目录类型", regex="^(drill|ground|diamagnetic)$"),
        file_name: str = Path(..., description="要删除的文件名")
):
    """
    删除指定目录下的文件，同时可转发到内部服务。
    """
    from urllib.parse import unquote
    file_name = unquote(file_name)  # 解决中文或特殊字符问题
    # 打印接收到的参数
    logger.info(f"🔹 directory_type: {directory_type}")
    logger.info(f"🔹 file_name (decoded): {file_name}")
    try:
        # 选择目标目录
        if directory_type == "drill":
            target_dir = DRILL_INPUT_DIR
        elif directory_type == "ground":
            target_dir = INPUT_DIR
        else:  # diamagnetic
            target_dir = DIAMAGNETIC_INPUT_DIR


        file_to_delete = target_dir / file_name

        if not file_to_delete.exists() or not file_to_delete.is_file():
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_to_delete}")

        # 删除本地文件
        file_to_delete.unlink()
        print(f"🗑️ 已删除文件: {file_to_delete}")



        return {
            "status": "success",
            "detail": f"文件 '{file_name}' 已从 '{directory_type}' 目录中删除"
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "detail": f"删除文件失败: {str(e)}"}

@app.get("/get_downloadable_files/{directory_type}")
def get_downloadable_files(
    directory_type: str = Path(..., description="要查询的目录类型", regex="^(drill|ground)$")
):
    """
    获取指定类型（drill/ground）对应的可下载文件列表。
    """
    try:
        # 根据类型选择对应的下载目录
        if directory_type == "drill":
            target_dir = DRILL_DOWNLOAD_DIR
        else:  # ground
            target_dir = GROUND_DOWNLOAD_DIR

        # 检查目录是否存在
        if not target_dir.exists() or not target_dir.is_dir():
            # 如果目录不存在，返回空列表，而不是报错，这样前端体验更好
            # 你也可以根据需要改为 raise HTTPException
            return {
                "status": "success",
                "directory": str(target_dir),
                "file_count": 0,
                "files": []
            }

        # 遍历目录，获取文件信息
        file_list = []
        for file_path in target_dir.iterdir():
            if file_path.is_file():
                file_list.append({
                    "file_name": file_path.name,
                    "size_bytes": file_path.stat().st_size
                })

        return {
            "status": "success",
            "directory": str(target_dir),
            "file_count": len(file_list),
            "files": file_list
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "detail": f"获取可下载文件列表失败: {str(e)}"}
# @app.delete("/delete_specific_file/{directory_type}/{file_name}")
# def delete_specific_file(
#         directory_type: str = Path(..., description="文件所在的目录类型", regex="^(drill|ground)$"),
#         file_name: str = Path(..., description="要删除的文件名")
# ):
#     """删除指定目录下的文件"""
#     try:
#         if directory_type == "drill":
#             target_dir = DRILL_INPUT_DIR
#         else:  # ground
#             target_dir = INPUT_DIR
#
#         file_to_delete = target_dir / file_name
#
#         if not file_to_delete.exists() or not file_to_delete.is_file():
#             raise HTTPException(status_code=404, detail=f"文件不存在: {file_to_delete}")
#
#         file_to_delete.unlink()
#         print(f"🗑️  已删除文件: {file_to_delete}")
#
#         return {
#             "status": "success",
#             "detail": f"文件 '{file_name}' 已从 '{directory_type}' 目录中成功删除。"
#         }
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         return {"status": "error", "detail": f"删除文件失败: {str(e)}"}
#
#
# # ... (确保你已经定义了这两个下载目录)
# # GROUND_DOWNLOAD_DIR = INDEX_ROOT / "ground_graph_fill"
# # DRILL_DOWNLOAD_DIR = DRILL_INDEX_ROOT / "drill_graph_fill"

@app.get("/download_file/{directory_type}/{file_name}")
def download_file(
    directory_type: str = Path(..., description="文件所在的目录类型", regex="^(drill|ground)$"),
    file_name: str = Path(..., description="要下载的文件名")
):
    """下载指定目录下的文件"""
    try:
        if directory_type == "drill":
            target_dir = DRILL_DOWNLOAD_DIR
        else:  # ground
            target_dir = GROUND_DOWNLOAD_DIR

        file_to_download = target_dir / file_name

        if not file_to_download.exists():
            # 返回 404 错误和 JSON 格式的错误信息
            raise HTTPException(status_code=404, detail=f"文件不存在: {file_to_download}")

        if not file_to_download.is_file():
            # 返回 400 错误和 JSON 格式的错误信息
            raise HTTPException(status_code=400, detail=f"'{file_name}' 不是一个文件。")

        print(f"📥 准备下载文件: {file_to_download}")

        return FileResponse(
            path=file_to_download,
            filename=file_name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        # 返回 500 错误和 JSON 格式的错误信息
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}") from e


