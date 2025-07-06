from docling.document_converter import DocumentConverter
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import PdfFormatOption, ImageFormatOption
import mimetypes
from datetime import datetime
import os
from huggingface_hub import snapshot_download
import cv2
from PIL import Image, ImageEnhance, ImageFilter


class DocumentProcessor:
    
    def __init__(self):
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.do_ocr = True 
        self.pipeline_options.do_table_structure = True 
        self.pipeline_options.table_structure_options.do_cell_matching = True
        
        download_path = snapshot_download(repo_id="SWHL/RapidOCR")
        det_model_path = os.path.join(download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx")
        rec_model_path = os.path.join(download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx")
        cls_model_path = os.path.join(download_path, "PP-OCRv3", "ch_ppocr_mobile_v2.0_cls_train.onnx")
        
        self.pipeline_options.ocr_options = RapidOcrOptions(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        cls_model_path=cls_model_path,
    )
        self.pipeline_options.table_structure_options.mode = "accurate"  
        self.pipeline_options.images_scale = 5.0
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options, backend=PyPdfiumDocumentBackend),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options= self.pipeline_options)
            }
        )
        
    def get_mime_type(self, filepath: str) -> str:
        mime_type, _ = mimetypes.guess_type(filepath)
        return mime_type
    
    def clean_table_data(self, df):

        df = df.dropna(how='all').dropna(axis=1, how='all')

        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                df[col] = df[col].str.replace(r'[^\w\s\-\.,;:()/%$]', '', regex=True)
                df[col] = df[col].replace(['nan', 'None', 'null'], '')
        
        return df
    
    def extract_text_from_file(self, filepath: str):
        mime_type = self.get_mime_type(filepath)
        if mime_type.startswith('image/'):
            try:
                processed_filepath = self.preprocess_image(filepath)
            except Exception as e:
                pass
        filename = os.path.basename(processed_filepath)
        
        if mime_type is None:
            raise ValueError("Could not determine MIME type of the file.")
        
        try:
            result = self.converter.convert(filepath)
            extracted_text = result.document.export_to_text()
            

            metadata = {
                "filename": filename,
                "mime_type": mime_type,
                "text": extracted_text,
                "processing_method": "Docling with Enhanced OCR",
                "timestamp": datetime.now().isoformat(),
                "word_count": len(extracted_text.split()) if extracted_text else 0,
                "page_count": len(result.document.pages) if hasattr(result.document, 'pages') else 1,
                "has_tables": bool(result.document.tables) if hasattr(result.document, 'tables') else False,
                "has_images": bool(result.document.pictures) if hasattr(result.document, 'pictures') else False
            }
            
            # Enhanced table processing
            if hasattr(result.document, 'tables') and result.document.tables:
                table_data = []
                for i, table in enumerate(result.document.tables):
                    try:
                        table_df = table.export_to_dataframe()
                        cleaned_df = self.clean_table_data(table_df)
                        if cleaned_df.empty:
                            print(f"Warning: Table {i+1} is empty after cleaning")
                            continue

                        print(f"--- Table {i+1} Raw Data ---")
                        print(table_df.to_string())
                        print(f"--- Table {i+1} Cleaned Data ---")
                        print(cleaned_df.to_string())
                        
                        # Store table information
                        table_info = {
                            "table_id": i + 1,
                            "csv_data": cleaned_df.to_csv(index=False),
                            "shape": cleaned_df.shape,
                            "confidence": getattr(table, 'confidence', None)
                        }
                        
                        table_data.append(table_info)
                        
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error processing table {i+1}: {e}")
                      
                metadata["tables"] = table_data
                metadata["table_count"] = len(table_data)
            
            # Enhanced image processing
            if hasattr(result.document, 'pictures') and result.document.pictures:
                image_info = []
                for i, picture in enumerate(result.document.pictures):
                    image_data = {
                        "image_id": i + 1,
                        "caption": getattr(picture, 'caption', None),
                        "confidence": getattr(picture, 'confidence', None)
                    }
                    image_info.append(image_data)
                
                metadata["images"] = image_info
                metadata["image_count"] = len(image_info)
            
            return metadata
            
        except Exception as e:
            raise ValueError(f"Error processing file with Docling: {e}")
        
        
    
    def preprocess_image(self, image_path: str, output_path: str = None) -> str:

        try:

            if isinstance(image_path, str):
                image = cv2.imread(image_path)
            else:
                image = image_path
            
            if image is None:
                raise ValueError("Could not load image")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            pil_img = Image.fromarray(thresh)
            
            sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            
            if output_path is None:
                output_path = image_path.replace('.', '_preprocessed.')
            
            sharpened.save(output_path)
            
            return output_path
            
        except Exception as e:
            return image_path