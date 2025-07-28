import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "tesseract"
os.environ["TESSDATA_PREFIX"] = "./tessdata"

def parse_date(date_str):
    if not date_str or len(date_str) != 6 or not date_str.isdigit():
        return ""
    year = int(date_str[:2]) + (2000 if int(date_str[:2]) < 50 else 1900)
    return f"{year}-{date_str[2:4]}-{date_str[4:6]}"

def parse_gender(gender_char):
    return {'M': 'Male', 'F': 'Female'}.get(gender_char, "Unspecified")

def parse_mrz(mrz_text):
    lines = [line.strip() for line in mrz_text.splitlines() if line.strip()]
    result = {
        "passport_type": "Unknown",
        "birth_date": "",
        "expiry_date": "",
        "given_names": "",
        "issuing_country": "",
        "nationality": "",
        "document_number": "",
        "raw": mrz_text.strip(),
        "sex": "",
        "surname": ""
    }

    cleaned_lines = [''.join(c for c in line if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<') for line in lines]

    if len(cleaned_lines) == 2 and all(len(line) >= 44 for line in cleaned_lines):
        line1, line2 = cleaned_lines
        result.update({
            "issuing_country": line1[2:5],
            "document_number": line2[0:9].replace("<", ""),
            "nationality": line2[10:13].replace("0", "O"),
            "birth_date": parse_date(line2[13:19]),
            "sex": parse_gender(line2[20]),
            "expiry_date": parse_date(line2[21:27])
        })
        name_parts = line1[5:].split("<<", 1)
        result["surname"] = name_parts[0].replace("<", " ").strip()
        if len(name_parts) > 1:
            result["given_names"] = " ".join(n.replace("<", " ").strip() for n in name_parts[1].split() if n.strip())

    elif len(cleaned_lines) == 3 and all(len(line) >= 30 for line in cleaned_lines):
        line1, line2, line3 = cleaned_lines
        result.update({
            "document_number": line1[5:14].replace("<", ""),
            "issuing_country": line1[2:5],
            "birth_date": parse_date(line2[0:6]),
            "sex": parse_gender(line2[7]),
            "expiry_date": parse_date(line2[8:14]),
            "nationality": line2[15:18].replace("0", "O")
        })
        name_parts = line3.split("<<", 1)
        result["surname"] = name_parts[0].replace("<", " ").strip()
        if len(name_parts) > 1:
            result["given_names"] = " ".join(n.replace("<", " ").strip() for n in name_parts[1].split() if n.strip())

    return result
