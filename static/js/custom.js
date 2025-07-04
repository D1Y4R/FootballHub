// Global fonksiyonlar

function showPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName, isRefresh = false) {
    console.log("showPrediction çağrıldı:", {homeTeamId, awayTeamId, homeTeamName, awayTeamName});
    
    // Ana showPrediction fonksiyonunu çağır
    if (typeof window.showPrediction !== 'undefined') {
        window.showPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName, isRefresh);
    } else {
        console.error("Global showPrediction fonksiyonu bulunamadı");
    }
}

function formatPrediction(type, prediction) {
    // main.js'deki formatPrediction fonksiyonuna göre güncellendi
    switch (type) {
        case 'over_2_5_goals':
        case '2.5_üst_alt':
            if (prediction === 'YES' || prediction === '2.5 ÜST') return '2.5 ÜST';
            if (prediction === 'NO' || prediction === '2.5 ALT') return '2.5 ALT';
            return prediction;
        case 'over_3_5_goals':
        case '3.5_üst_alt':
            if (prediction === 'YES' || prediction === '3.5 ÜST') return '3.5 ÜST';
            if (prediction === 'NO' || prediction === '3.5 ALT') return '3.5 ALT';
            return prediction;
        case 'both_teams_to_score':
        case 'kg_var_yok':
            if (prediction === 'YES' || prediction === 'KG VAR') return 'KG VAR';
            if (prediction === 'NO' || prediction === 'KG YOK') return 'KG YOK';
            return prediction;
        case 'match_result':
        case 'maç_sonucu':
            switch(prediction) {
                case 'HOME_WIN': return 'MS1';
                case 'DRAW': return 'X';
                case 'AWAY_WIN': return 'MS2';
                case 'MS1': return 'MS1';
                case 'X': return 'X';
                case 'MS2': return 'MS2';
                default: return prediction;
            }
        default:
            // Genel YES/NO değerlerini çevir
            if (prediction === 'YES') return 'VAR';
            if (prediction === 'NO') return 'YOK';
            return prediction;
    }
}

function refreshPrediction(homeTeamId, awayTeamId, homeTeamName, awayTeamName) {
    // Yükleniyor göster, içeriği gizle
    $('#predictionLoading').show();
    $('#predictionContent').hide();
    $('#predictionError').hide();
    
    // API isteği yap - force_update parametresi true olarak gönder
    const url = `/api/predict-match/${homeTeamId}/${awayTeamId}?home_name=${encodeURIComponent(homeTeamName)}&away_name=${encodeURIComponent(awayTeamName)}&force_update=true`;
    
    $.ajax({
        url: url,
        type: 'GET',
        dataType: 'json',
        success: function(data) {
            // Başarılı güncelleme sonrası verileri göster
            window.displayPredictionData(data, homeTeamName, awayTeamName);
        },
        error: function(error) {
            $('#predictionLoading').hide();
            $('#predictionError').text('Tahmin güncellenirken hata oluştu: ' + error.responseJSON?.error || 'Sunucu hatası').show();
        }
    });
}

// Sürpriz butonu işlevselliği tamamen kaldırıldı