# Functions to interact with the database
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, Patient
import pydicom as dicom
import glob

DATABASE = "./Backend/Databases/users.db"
PATIENTS_DATABASE = "./Backend/Databases/patients.db"


def get_db_connection(database):
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    return conn

def init__users_db():
    conn = get_db_connection(DATABASE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT NOT NULL,
            surname TEXT NOT NULL,
            is_doctor BOOLEAN NOT NULL
        )
    """)
    conn.commit()
    # Aggiungi un utente di esempio
    try:
        user = User(
            email="admin@admin.it",
            password="admin",
            name="Mario",
            surname="Rossi",
            is_doctor=True
        )
        insert_user(conn, user)
        print("User inserted successfully.")
    except sqlite3.IntegrityError:
        print("User already exists, skipping insertion.")
    except Exception as e:
        print(f"Error inserting user: {e}")

def init_patients_db():
    conn = sqlite3.connect(PATIENTS_DATABASE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY NOT NULL,
            name TEXT NOT NULL,
            sex TEXT NOT NULL,
            study_date TEXT NOT NULL,
            study_time TEXT NOT NULL,
            study_type TEXT NOT NULL,
            study_zone TEXT NOT NULL,
            study_result TEXT NOT NULL,
            image_height INTEGER NOT NULL,
            image_width INTEGER NOT NULL,
            image BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    # Aggiungi i pazienti di esempio
    data_dir = "./Backend/T4R-Studies"
    for item in glob.glob(data_dir + '/Breast-US-*.dcm'):
        patient = Patient.from_dicom(dicom_file=item)
        try:
            insert_patient(conn, patient)
            print("Patient inserted successfully.")
        except sqlite3.IntegrityError:
            print("Patient already exists, skipping insertion.")
        except Exception as e:
            print(f"Error inserting patient: {e}")
    conn.close()



def insert_user(conn, user):
    """
    Insert a new user into the database.
    """
    try:
        conn = get_db_connection(DATABASE)
        conn.execute(
            "INSERT INTO users (email, password, name, surname, is_doctor) VALUES (?, ?, ?, ?, ?)",
            (user.email, generate_password_hash(user.password), user.name, user.surname, user.is_doctor))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        print("User already exists, skipping insertion.")
        conn.close()
    except Exception as e:
        print(f"Error inserting user: {e}")
        conn.close()
    finally:
        conn.close()
    
    

def get_user_by_email(email):
    """
    Get a user by email.
    """
    conn = get_db_connection(DATABASE)
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    ).fetchone()
    conn.close()
    if user:
        return User(user["email"], user["password"], user["name"], user["surname"], user["is_doctor"], id = user["id"])
    return None

def insert_patient(conn, patient):
    """
    Insert a new patient into the database.
    """
    try:
        conn = get_db_connection(PATIENTS_DATABASE)
        conn.execute(
            "INSERT INTO patients (id, name, sex, study_date, study_time, study_type, study_zone, study_result, image_height, image_width, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (patient.id, patient.name, patient.sex, patient.study_date, patient.study_time, patient.study_type, patient.study_zone, patient.study_result, patient.image_height, patient.image_width, patient.image))
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        print("Patient already exists, skipping insertion.")
        conn.close()
    except Exception as e:
        print(f"Error inserting patient: {e}")
        conn.close()
    finally:
        conn.close()

def get_patients():
    """
    Get all patients from the database.
    """
    conn = get_db_connection(PATIENTS_DATABASE)
    patients = conn.execute(
        "SELECT * FROM patients"
    ).fetchall()
    conn.close()
    return [Patient(patient["id"], patient["name"], patient["sex"], patient["study_date"], patient["study_time"], patient["study_type"], patient["study_zone"], patient["study_result"], patient["image_height"], patient["image_width"], patient["image"]) for patient in patients]

def get_patient_by_id(patient_id):
    """
    Get a patient by ID.
    """
    conn = get_db_connection(PATIENTS_DATABASE)
    patient = conn.execute(
        "SELECT * FROM patients WHERE id = ?",
        (patient_id,)
    ).fetchone()
    conn.close()
    if patient:
        return Patient(patient["id"], patient["name"], patient["sex"], patient["study_date"], patient["study_time"], patient["study_type"], patient["study_zone"], patient["study_result"], patient["image_height"], patient["image_width"], patient["image"])
    return None