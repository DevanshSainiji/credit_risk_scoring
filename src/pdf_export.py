from fpdf import FPDF
import io

def generate_pdf_report(report_text: str) -> bytes:
    """
    Takes the markdown report text and converts it to a standard PDF byte stream
    which can be downloaded directly from Streamlit.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Intelligent Credit Risk Assessment Report", ln=True, align="C")
    pdf.ln(5)
    
    # Body text
    pdf.set_font("Arial", size=11)
    
    # Clean some basic markdown for fpdf (simple fallback)
    clean_text = report_text.replace("## ", "\n*** ").replace("**", "")
    
    import textwrap
    
    for line in clean_text.split("\n"):
        safe_line = line.encode('latin-1', 'replace').decode('latin-1')
        
        if not safe_line.strip():
            pdf.ln(7)
            continue
            
        wrapped_lines = textwrap.wrap(safe_line, width=90, break_long_words=True, replace_whitespace=False)
        
        for w_line in wrapped_lines:
            # Bypass multi_cell completely and force it to just print exactly this string on its own line
            pdf.cell(0, 7, txt=w_line, ln=True)
    
    # Output to byte stream instead of file so we can download immediately
    # fpdf2 returns bytearray, older fpdf returns str. We handle both cleanly.
    out = pdf.output(dest='S')
    pdf_bytes = out.encode('latin-1') if isinstance(out, str) else bytes(out)
    
    return pdf_bytes
