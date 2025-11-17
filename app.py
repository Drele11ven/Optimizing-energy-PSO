import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# عنوان
st.title("بهینه‌سازی مصرف انرژی ساختمان با PSO")

# بخش پارامترها
st.sidebar.header("پارامترهای PSO")

num_particles = st.sidebar.slider("تعداد ذرات", 5, 100, 30)
iterations = st.sidebar.slider("تعداد تکرارها", 10, 300, 100)
w = st.sidebar.slider("وزن اینرسی (w)", 0.1, 1.0, 0.7)
c1 = st.sidebar.slider("ضریب شناختی (c1)", 0.1, 2.5, 1.4)
c2 = st.sidebar.slider("ضریب اجتماعی (c2)", 0.1, 2.5, 1.4)

st.write("پارامترها تنظیم شد. برای اجرای الگوریتم آماده‌ای؟")
run_button = st.button("اجرای PSO")

if run_button:
    st.write("🔄 اجرای الگوریتم... (بعداً اینجا PSO را اضافه می‌کنیم)")
