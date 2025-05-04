# Models for the database
import pydicom

class User:
    def __init__(self, email, password, name, surname, is_doctor, id = None):
        self.id = id
        self.email = email
        self.password = password
        self.name = name
        self.surname = surname
        self.is_doctor = is_doctor

    def __repr__(self):
        return f"User({self.id}, {self.email}, {self.name}, {self.surname}, {self.is_doctor})"
    
class Patient:
    def __init__(self, id, name, sex, study_date, study_time, study_type, study_zone, study_result, image_height, image_width, image):
        self.id = id
        self.name = name
        self.sex = sex
        self.study_date = study_date
        self.study_time= study_time
        self.study_type = study_type
        self.study_zone = study_zone
        self.study_result = study_result
        self.image_height = image_height
        self.image_width = image_width
        self.image = image
    
    def __repr__(self):
        return f"Patient({self.id}, {self.name}, {self.sex}, {self.study_date}, {self.study_time}, {self.study_type}, {self.study_zone}, {self.study_result})"
    
    @classmethod
    def from_dicom(cls, dicom_file):
        """
        Reads a DICOM file and returns a Patient instance.
        """
        try:
            ds = pydicom.dcmread(dicom_file)
            id = str(ds.PatientID)
            name = str(ds.PatientName) if ds.PatientName != "" else "Unknown"
            sex = str(ds.PatientSex)
            study_date = str(ds.StudyDate[:4] + '-' + ds.StudyDate[4:6] + '-' + ds.StudyDate[6:])
            study_time = str(ds.StudyTime[:2] + ':' + ds.StudyTime[2:4] + ':' + ds.StudyTime[4:])
            description = [str(x) for x in ds.ClinicalTrialSeriesDescription.split(';')]
            study_type = description[1]
            study_zone = description[0]
            study_result = str(ds.ClinicalTrialSeriesID.split(';')[2])
            image_height = int(ds.Rows)
            image_width = int(ds.Columns)
            image = ds.pixel_array.tobytes()  # Store as bytes
            return cls(id, name, sex, study_date, study_time, study_type, study_zone, study_result, image_height, image_width, image)
        except Exception as e:
            print(f"Error reading DICOM file: {e}")
            return None
