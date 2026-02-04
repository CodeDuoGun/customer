"""
Milvus data insertion script supporting full-text search, vector search, and hybrid search.
"""
import argparse
import os
import pandas as pd
from typing import List, Dict, Any
from customer.rag.index_builder import IndexBuilder
from customer.utils.logger import logger
from customer.rag.milvus_processor import milvus_processor
from customer.config.config import config
from customer.service.embeddings import DoubaoEmbeddings


def load_and_split_docx(file_path: str, chunk_size: int = 500, chunk_overlap: int = 10) -> List[Dict[str, Any]]:
    """
    Load and split docx file into chunks.

    Args:
        file_path: Path to docx file
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    try:
        # This is a placeholder - you need to implement actual docx loading
        # For now, return dummy data
        index_builder = IndexBuilder()
        documents = index_builder.build_from_files(["customer/data/AI智能客服知识库5-6.docx"])
        chunks = index_builder.process_documents(documents)
        logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks
    except Exception as e:
        logger.error(f"Failed to load docx file {file_path}: {str(e)}")
        return []


def gen_qa_by_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Generate QA pairs from file.

    Args:
        file_path: Path to QA file

    Returns:
        List of QA pairs
    """
    try:
        qa_pairs = []
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                qa_pairs.append({
                    "question": str(row.get("question", "")),
                    "answer": str(row.get("answer", "")),
                    "category": str(row.get("category", "")),
                    "id": str(row.get("id", hash(str(row.get("question", "")))))
                })
        logger.info(f"Generated {len(qa_pairs)} QA pairs from {file_path}")
        return qa_pairs
    except Exception as e:
        logger.error(f"Failed to generate QA pairs from {file_path}: {str(e)}")
        return []


def test_vector_search():
    """Test vector search functionality."""
    try:
        collection_name = f"{config.env_version}_doctor"
        query = "心脏病专家"

        logger.info(f"Testing vector search on collection: {collection_name}")

        # Test vector search
        results = milvus_processor.vector_search(
            collection_name=collection_name,
            query=query,
            limit=5
        )

        for i, result in enumerate(results, 1):
            logger.info(f"{i}. Score: {result['score']}, Entity: {result['entity']}")

    except Exception as e:
        logger.error(f"Vector search test failed: {str(e)}")


def main():
    """Main function for data insertion."""
    # Initialize embedding model
    embed_model = DoubaoEmbeddings()

    parser = argparse.ArgumentParser(description="Milvus数据操作")
    parser.add_argument("--index_name", type=str, default="base_qa", help="集合名称")
    parser.add_argument("--host", type=str, default=config.MILVUS_HOST, help="Milvus主机地址")
    parser.add_argument("--port", type=str, default=config.MILVUS_PORT, help="Milvus端口")
    parser.add_argument("--embed_model", type=str, default="doubao", help="embedding 模型")
    parser.add_argument("--alpha", type=bool, default=True, help="是否为本地使用")

    args = parser.parse_args()

    # Update processor connection if custom host/port provided
    if args.host != config.MILVUS_HOST or args.port != config.MILVUS_PORT:
        milvus_processor.disconnect()
        milvus_processor.host = args.host
        milvus_processor.port = args.port
        milvus_processor._connect()

    try:
        collection_name = f"{config.env_version}_{args.index_name}"

        if args.index_name in ["doc_qa", "test_word"]:
            # Document chunks insertion
            print("Loading document chunks...")

            file_path = f"customer/data/AI智能客服知识库5-6.docx"
            chunks = load_and_split_docx(file_path, chunk_size=500, chunk_overlap=10)

            if not chunks:
                logger.error("No chunks loaded, exiting")
                return

            # Create collection schema
            schema_config = {
                "id": {"type": "VARCHAR", "max_length": 100, "is_primary": True},
                "content": {"type": "VARCHAR", "max_length": 65535},
                "vector": {"type": "FLOAT_VECTOR", "dim": 4096 if args.embed_model == "doubao" else 1024},
                "metadata": {"type": "JSON"}
            }

            milvus_processor.create_collection(collection_name, schema_config, "Document chunks collection")

            # Insert documents
            milvus_processor.insert_documents(collection_name, chunks, text_field="content", vector_field="vector")

        elif args.index_name in ["test_qa", "qa"]:
            # QA pairs insertion
            print("Loading QA pairs...")

            file_path = f"customer/data/qa_out_final_{'0526' if args.index_name == 'test_qa' else '0611'}.csv"
            qa_pairs = gen_qa_by_file(file_path)

            if not qa_pairs:
                logger.error("No QA pairs loaded, exiting")
                return

            # Create collection schema
            schema_config = {
                "id": {"type": "VARCHAR", "max_length": 100, "is_primary": True},
                "kw_question": {"type": "VARCHAR", "max_length": 1000},
                "question": {"type": "VARCHAR", "max_length": 65535},
                "q_vector": {"type": "FLOAT_VECTOR", "dim": 4096 if args.embed_model == "doubao" else 1024},
                "answer": {"type": "VARCHAR", "max_length": 65535},
                "answer_vector": {"type": "FLOAT_VECTOR", "dim": 4096 if args.embed_model == "doubao" else 1024},
                "category": {"type": "VARCHAR", "max_length": 500}
            }

            milvus_processor.create_collection(collection_name, schema_config, "QA pairs collection")

            # Insert QA pairs
            milvus_processor.insert_qa_pairs(collection_name, qa_pairs, embed_model=embed_model)

        elif args.index_name == "doctor":
            # Doctor information insertion
            print("Loading doctor information...")

            # Create collection schema
            schema_config = {
                "id": {"type": "VARCHAR", "max_length": 100, "is_primary": True},
                "序号": {"type": "INT64"},
                "姓名": {"type": "VARCHAR", "max_length": 100},
                "ID": {"type": "VARCHAR", "max_length": 100},
                "所在区域": {"type": "VARCHAR", "max_length": 500},
                "特殊称号": {"type": "VARCHAR", "max_length": 200},
                "执业医院": {"type": "VARCHAR", "max_length": 500},
                "goodvector": {"type": "FLOAT_VECTOR", "dim": 4096 if args.embed_model == "doubao" else 1024},
                "简介": {"type": "VARCHAR", "max_length": 65535},
                "擅长": {"type": "VARCHAR", "max_length": 65535},
                "出诊地点": {"type": "VARCHAR", "max_length": 500},
                "vector": {"type": "FLOAT_VECTOR", "dim": 4096 if args.embed_model == "doubao" else 1024}
            }

            milvus_processor.create_collection(collection_name, schema_config, "Doctor information collection")

            # Load and insert doctor data
            excel_file = "customer/data/医师id_20250922.xlsx"
            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file)
                doctor_data = []

                for _, row in df.iterrows():
                    doctor_dict = {}
                    for col in df.columns:
                        doctor_dict[col] = str(row[col]) if pd.notna(row[col]) else ""

                    # Generate embeddings for relevant fields
                    text_content = f"{doctor_dict.get('擅长', '')} {doctor_dict.get('简介', '')}".strip()
                    if text_content:
                        embedding = embed_model.embed_query(text_content)
                        doctor_dict["goodvector"] = embedding
                        doctor_dict["vector"] = embedding

                    doctor_dict["id"] = doctor_dict.get("ID", str(hash(str(doctor_dict))))
                    doctor_data.append(doctor_dict)

                milvus_processor.insert_documents(collection_name, doctor_data, text_field="擅长", vector_field="goodvector")
            else:
                logger.error(f"Doctor data file not found: {excel_file}")

        elif args.index_name == "doctor_info":
            # Detailed doctor information insertion
            print("Loading detailed doctor information...")

            # Create collection schema
            schema_config = {
                "id": {"type": "VARCHAR", "max_length": 100, "is_primary": True},
                "序号": {"type": "INT64"},
                "姓名": {"type": "VARCHAR", "max_length": 100},
                "ID": {"type": "VARCHAR", "max_length": 100},
                "性别": {"type": "VARCHAR", "max_length": 10},
                "出生年份": {"type": "VARCHAR", "max_length": 10},
                "年龄": {"type": "INT64"},
                "所在区域": {"type": "VARCHAR", "max_length": 500},
                "职称": {"type": "VARCHAR", "max_length": 200},
                "职位": {"type": "VARCHAR", "max_length": 200},
                "擅长病种": {"type": "VARCHAR", "max_length": 65535},
                "goodvector": {"type": "FLOAT_VECTOR", "dim": 4096 if args.embed_model == "doubao" else 1024},
                "荣誉头衔": {"type": "VARCHAR", "max_length": 65535},
                "主要成就": {"type": "VARCHAR", "max_length": 65535},
                "教育背景": {"type": "VARCHAR", "max_length": 65535},
                "学历": {"type": "VARCHAR", "max_length": 100},
                "抖音名称": {"type": "VARCHAR", "max_length": 200},
                "快手名称": {"type": "VARCHAR", "max_length": 200},
                "社会职务": {"type": "VARCHAR", "max_length": 65535},
                "挂号费": {"type": "VARCHAR", "max_length": 100},
                "出诊时间": {"type": "VARCHAR", "max_length": 500},
                "处方单价": {"type": "VARCHAR", "max_length": 100},
                "治疗特色": {"type": "VARCHAR", "max_length": 65535},
                "治疗案例1": {"type": "VARCHAR", "max_length": 65535},
                "治疗案例2": {"type": "VARCHAR", "max_length": 65535}
            }

            milvus_processor.create_collection(collection_name, schema_config, "Detailed doctor information collection")

            # Load and insert detailed doctor data
            excel_file = "customer/data/doctor_info_0409.xlsx"
            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file)
                doctor_data = []

                for _, row in df.iterrows():
                    doctor_dict = {}
                    for col in df.columns:
                        if col == "年龄":
                            doctor_dict[col] = int(row[col]) if pd.notna(row[col]) else 0
                        else:
                            doctor_dict[col] = str(row[col]) if pd.notna(row[col]) else ""

                    # Generate embeddings for treatment specialty
                    specialty = doctor_dict.get("治疗特色", "")
                    if specialty:
                        embedding = embed_model.embed_query(specialty)
                        doctor_dict["goodvector"] = embedding

                    doctor_dict["id"] = doctor_dict.get("ID", str(hash(str(doctor_dict))))
                    doctor_data.append(doctor_dict)

                milvus_processor.insert_documents(collection_name, doctor_data, text_field="治疗特色", vector_field="goodvector")
            else:
                logger.error(f"Detailed doctor data file not found: {excel_file}")

        elif args.index_name in ["primary_disease", "secondary_disease"]:
            # Disease information insertion
            print(f"Loading {args.index_name} information...")

            if args.index_name == "primary_disease":
                schema_config = {
                    "id": {"type": "VARCHAR", "max_length": 100, "is_primary": True},
                    "primary_disease": {"type": "VARCHAR", "max_length": 500},
                    "doctors": {"type": "VARCHAR", "max_length": 65535},
                    "primary_disease_vector": {"type": "FLOAT_VECTOR", "dim": 4096 if args.embed_model == "doubao" else 1024}
                }
            else:
                schema_config = {
                    "id": {"type": "VARCHAR", "max_length": 100, "is_primary": True},
                    "secondary_disease": {"type": "VARCHAR", "max_length": 500},
                    "doctors": {"type": "VARCHAR", "max_length": 65535},
                    "kw_secondary_disease": {"type": "VARCHAR", "max_length": 500}
                }

            milvus_processor.create_collection(collection_name, schema_config, f"{args.index_name} information collection")

            # Load and insert disease data
            excel_file = "customer/data/四惠医疗互联网医院疾病库-20250508-V3.xlsx"
            sheet_name = "二级病种" if args.index_name == "secondary_disease" else "一级病种"

            if os.path.exists(excel_file):
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                disease_data = []

                for _, row in df.iterrows():
                    disease_dict = {}
                    for col in df.columns:
                        disease_dict[col] = str(row[col]) if pd.notna(row[col]) else ""

                    disease_dict["id"] = str(hash(str(disease_dict)))

                    # Generate embeddings for primary diseases
                    if args.index_name == "primary_disease":
                        disease_name = disease_dict.get("primary_disease", "")
                        if disease_name:
                            embedding = embed_model.embed_query(disease_name)
                            disease_dict["primary_disease_vector"] = embedding

                    disease_data.append(disease_dict)

                milvus_processor.insert_documents(collection_name, disease_data,
                                               text_field="primary_disease" if args.index_name == "primary_disease" else "secondary_disease",
                                               vector_field="primary_disease_vector" if args.index_name == "primary_disease" else None)
            else:
                logger.error(f"Disease data file not found: {excel_file}")

        elif args.index_name == "test_vector_search":
            # Test vector search functionality
            test_vector_search()

        else:
            logger.error(f"Unknown index_name: {args.index_name}")
            logger.info("Supported index names: base_qa, test_word, test_qa, qa, doctor, doctor_info, primary_disease, secondary_disease, test_vector_search")

        logger.info(f"Data insertion completed for collection: {collection_name}")

    except Exception as e:
        logger.error(f"Data insertion failed: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Disconnect from Milvus
        milvus_processor.disconnect()


if __name__ == "__main__":
    main()
