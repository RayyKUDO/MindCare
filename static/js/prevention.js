// Load prevention plan based on risk from session data
function loadPreventionPlan() {
    // Get predictions from window object
    let predictions = window.predictions || [];

    const riskAssessmentDiv = document.getElementById('riskAssessment');
    const preventionPlanDiv = document.getElementById('preventionPlan');

    if (!riskAssessmentDiv || !preventionPlanDiv) {
        console.error('Required elements not found');
        return;
    }

    if (predictions.length === 0) {
        // Hide risk assessment and personalized plan if no predictions
        riskAssessmentDiv.style.display = 'none';
        preventionPlanDiv.style.display = 'none';
        return;
    }

    // Show sections if predictions exist
    riskAssessmentDiv.style.display = 'block';
    preventionPlanDiv.style.display = 'block';

    const latestPrediction = predictions[predictions.length - 1];
    const risk = latestPrediction.risk;
    const probability = latestPrediction.probability;

    // Update risk assessment
    document.getElementById('riskAssessment').innerHTML = `
        <div class="mb-4">
            <span class="inline-block px-4 py-2 rounded-full text-white font-semibold text-lg ${
                risk === 'Rendah' ? 'bg-green-500' :
                risk === 'Sedang' ? 'bg-yellow-500' : 'bg-red-500'
            }">Risiko ${risk} (${probability}%)</span>
        </div>
        <p class="text-gray-600">Rencana pencegahan disesuaikan berdasarkan profil risiko Anda</p>
    `;

    // Generate personalized plan
    let planContent = '';
    if (risk === 'Rendah') {
        planContent = `
            <div class="bg-green-50 p-6 rounded-lg mb-4">
                <h3 class="text-lg font-semibold text-green-800 mb-3">Fokus Pencegahan Dasar</h3>
                <ul class="space-y-2 text-green-700">
                    <li>Pertahankan gaya hidup sehat dengan olahraga teratur. seperti, exercise aerobic selama 30 menit/hari </li>
                    <li>Pertahankan pola makan yang sehat dan minum vitamin yang kaya akan antioksidan </li>
                    <li>Lakukan stimulasi kognitif minimal 3-4 kali seminggu</li>
                    <li>Monitor kesehatan secara rutin setiap 6 bulan</li>
                </ul>
            </div>
        `;
    } else if (risk === 'Sedang') {
        planContent = `
            <div class="bg-yellow-50 p-6 rounded-lg mb-4">
                <h3 class="text-lg font-semibold text-yellow-800 mb-3">Pencegahan Intensif</h3>
                <ul class="space-y-2 text-yellow-700">
                    <li>Tingkatkan aktivitas fisik dan olahraga. seperti, exercise aerobic selama 30 menit/hari </li>
                    <li>Fokus pada pola makan yang sehat dengan cara konsumsi sayuran, kacang-kacangan, buah-buahan, dan khususnya rendah lemak berbahaya, pemanis, dan karbohidrat olahan. </li>
                    <li>Kelola stress dengan melakukan meditasi, relaksasi progresif, dan latihan pernafasan dengan total 1 jam per hari. terdapat altenatif lain yaitu mendengarkan musik menenangkan sebagai bantuan meditasi dan insomnia</li>
                    <li>Konsultasi dokter untuk pemeriksaan kesehatan menyeluruh</li>
                </ul>
            </div>
        `;
    } else {
        planContent = `
            <div class="bg-red-50 p-6 rounded-lg mb-4">
                <h3 class="text-lg font-semibold text-red-800 mb-3">Pencegahan Prioritas Tinggi</h3>
                <ul class="space-y-2 text-red-700 mb-4">
                    <li>Segera konsultasi dengan spesialis neurologi</li>
                    <li>Kontrol faktor risiko seperti hipertensi dan diabetes</li>
                    <li>Lakukan MRI otak dan tes kognitif menyeluruh</li>
                    <li>Terapi kombinasi: konsumsi obat sesuai anjuran dokter, diet sehat seperti diet mediterania, diet vegan, diet dash</li>
                    <li>Dukungan keluarga dan mengikuti komunitas dukungan selama satu jam per sesi yang dipandu oleh profesional kesehatan mental</li>
                </ul>
                <div class="border-t pt-4">
                    <h4 class="font-semibold mb-2 text-red-800">Rekomendasi Dokter & Rumah Sakit</h4>
                    <ul class="list-disc list-inside text-red-700 space-y-1">
                        <li>Konsultasi dengan Dokter Spesialis Neurologi</li>
                        <li>Konsultasi dengan Dokter Spesialis Geriatri/Lansia</li>
                        <li>Konsultasi dengan Dokter Spesialis Ahli Kejiwaan, jika terdapat gejala yang dialami berpengaruh ke mental dan tingkah laku</li>
                        <li>Lakukan MRI otak dan tes darah lengkap</li>
                    </ul>
                </div>
            </div>
        `;
    }

    document.getElementById('preventionPlan').innerHTML = planContent;
}

// Load data on page load
document.addEventListener('DOMContentLoaded', function() {
    loadPreventionPlan();
});
