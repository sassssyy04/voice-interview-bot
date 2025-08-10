class VoiceBot {
    constructor() {
        this.candidateId = null;
        this.isRecording = false;
        this.isPaused = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.conversationActive = false;
        this.lastTurnStartTime = null;
        this.minRecordingMs = 300;
        this.recordingStartTimeMs = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.checkBrowserSupport();
    }

    initializeElements() {
        // UI Elements
        this.micButton = document.getElementById('micButton');
        this.micStatus = document.getElementById('micStatus');
        this.startBtn = document.getElementById('startBtn');
        this.demoBtn = document.getElementById('demoBtn');
        this.pauseBtn = document.getElementById('pauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.audioVisualizer = document.getElementById('audioVisualizer');
        this.progressBar = document.getElementById('progressBar');
        this.progressText = document.getElementById('progressText');
        this.conversationLog = document.getElementById('conversationLog');
        this.responseAudio = document.getElementById('responseAudio');
        this.connectionStatus = document.getElementById('connection-status');
        
        // Metrics
        this.responseTimeEl = document.getElementById('responseTime');
        this.voiceConfidenceEl = document.getElementById('voiceConfidence');
        this.fieldsCollectedEl = document.getElementById('fieldsCollected');
        
        // Profile Elements
        this.candidateProfile = document.getElementById('candidateProfile');
        this.profileElements = {
            pincode: document.getElementById('profilePincode'),
            salary: document.getElementById('profileSalary'),
            shift: document.getElementById('profileShift'),
            languages: document.getElementById('profileLanguages'),
            twoWheeler: document.getElementById('profileTwoWheeler'),
            experience: document.getElementById('profileExperience')
        };
        
        // Roles Preview (disabled)
        this.rolesPreview = document.getElementById('rolesPreview');
        this.rolesPreviewList = document.getElementById('rolesPreviewList');
        if (this.rolesPreview) { this.rolesPreview.classList.add('hidden'); }
        
        // Job matches
        this.jobMatches = document.getElementById('jobMatches');
        this.jobMatchesList = document.getElementById('jobMatchesList');
    }

    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startConversation());
        this.demoBtn.addEventListener('click', () => this.runJobMatchingDemo());
        this.pauseBtn.addEventListener('click', () => this.pauseConversation());
        this.stopBtn.addEventListener('click', () => this.stopConversation());
        // this.micButton.addEventListener('click', () => this.toggleRecording());
        // ... existing code ...
        this.micButton.addEventListener('mousedown', () => this.onPressStart());
        this.micButton.addEventListener('touchstart', () => this.onPressStart(), { passive: true });
        document.addEventListener('mouseup', () => this.onPressEnd());
        document.addEventListener('touchend', () => this.onPressEnd());
        document.addEventListener('touchcancel', () => this.onPressEnd());
        
        // Audio playback ended - do not auto-start listening in push-to-talk mode
        this.responseAudio.addEventListener('ended', () => {
            if (this.conversationActive && !this.isPaused) {
                this.micStatus.textContent = 'Press and hold to talk';
            }
        });
    }

    checkBrowserSupport() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showError('Your browser does not support audio recording. Please use Chrome, Firefox, or Safari.');
            return false;
        }
        
        if (!window.MediaRecorder) {
            this.showError('MediaRecorder is not supported in your browser.');
            return false;
        }
        
        return true;
    }

    async startConversation() {
        try {
            this.updateConnectionStatus('Starting...', 'text-yellow-500');
            
            // Request microphone permission
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            // Start conversation with backend
            const response = await fetch('/api/v1/conversation/start', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to start conversation');
            }
            
            const data = await response.json();
            this.candidateId = data.candidate_id;
            
            // Check if we're in demo mode
            if (data.audio_format === "text" || !data.audio_data) {
                // Demo mode - show message instead of playing audio
                this.addToConversationLog('Bot', data.message, false);
                this.micStatus.textContent = 'Demo Mode: Enable Google Cloud APIs for voice';
            } else {
                // Play initial greeting
                await this.playAudioFromBase64(data.audio_data);
                this.addToConversationLog('Bot', 'Namaste! Main aapka voice assistant hun job interview ke liye।', true);
            }
            
            // Update UI
            this.conversationActive = true;
            this.updateUIForActiveConversation();
            this.updateConnectionStatus('Connected', 'text-green-500');
            this.addToConversationLog('Bot', 'Namaste! Main aapka voice assistant hun job interview ke liye।', true);
            this.candidateProfile.classList.remove('hidden');
            this.micStatus.textContent = 'Press and hold to talk';
            
            // Start listening after greeting (disabled for push-to-talk)
            // setTimeout(() => this.startListening(), 1000);
            
        } catch (error) {
            console.error('Error starting conversation:', error);
            this.showError('Failed to start conversation: ' + error.message);
            this.updateConnectionStatus('Error', 'text-red-500');
        }
    }

    pauseConversation() {
        this.isPaused = !this.isPaused;
        
        if (this.isPaused) {
            this.stopRecording();
            this.pauseBtn.innerHTML = '<i class="fas fa-play mr-2"></i>Resume';
            this.micStatus.textContent = 'Conversation paused';
            this.updateConnectionStatus('Paused', 'text-yellow-500');
        } else {
            this.pauseBtn.innerHTML = '<i class="fas fa-pause mr-2"></i>Pause';
            this.updateConnectionStatus('Connected', 'text-green-500');
            // In push-to-talk mode, do not auto-start listening
            this.micStatus.textContent = 'Press and hold to talk';
        }
    }

    stopConversation() {
        this.conversationActive = false;
        this.isPaused = false;
        this.stopRecording();
        
        // Reset UI
        this.updateUIForInactiveConversation();
        this.updateConnectionStatus('Disconnected', 'text-gray-500');
        this.micStatus.textContent = 'Click to start conversation';
        this.audioVisualizer.classList.add('hidden');
        
        // Roles preview disabled; no job matches fetch
    }

    async startListening() {
        if (!this.conversationActive || this.isPaused || this.isRecording) return;
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            this.audioChunks = [];
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.processRecording();
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.isRecording = true;
            this.recordingStartTimeMs = Date.now();
            this.mediaRecorder.start();
            
            this.updateRecordingUI(true);
            
            // Auto-stop recording after 10 seconds (disabled for push-to-talk)
            // setTimeout(() => {
            //     if (this.isRecording) {
            //         this.stopRecording();
            //     }
            // }, 10000);
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.showError('Could not access microphone: ' + error.message);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.updateRecordingUI(false);
        }
    }

    toggleRecording() {
        if (!this.conversationActive) {
            this.startConversation();
        } else if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startListening();
        }
    }

    // Start recording on press, stop on release (press-to-talk)
    onPressStart() {
        if (!this.conversationActive) {
            this.startConversation();
            return;
        }
        if (this.isPaused) return;
        // Barge-in: stop TTS playback when user starts speaking
        if (this.responseAudio && !this.responseAudio.paused) {
            try {
                this.responseAudio.pause();
                this.responseAudio.currentTime = 0;
            } catch (e) {
                // no-op
            }
        }
        if (!this.isRecording) {
            this.startListening();
        }
    }

    onPressEnd() {
        if (this.isRecording) {
            this.stopRecording();
        }
    }

    async processRecording() {
        if (this.audioChunks.length === 0) return;
        
        try {
            const durationMs = this.recordingStartTimeMs ? (Date.now() - this.recordingStartTimeMs) : 0;
            if (durationMs < this.minRecordingMs) {
                // Too short; ignore and prompt user to hold longer
                this.audioChunks = [];
                if (this.conversationActive) {
                    this.micStatus.textContent = 'Hold longer to record';
                    setTimeout(() => {
                        if (this.conversationActive && !this.isRecording) {
                            this.micStatus.textContent = 'Press and hold to talk';
                        }
                    }, 1000);
                }
                return;
            }
            
            this.micStatus.textContent = 'Processing your voice...';
            
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            // Skip if blob is suspiciously small (e.g., silence or accidental tap)
            if (audioBlob.size < 500) {
                this.audioChunks = [];
                if (this.conversationActive) {
                    this.micStatus.textContent = 'Hold longer to record';
                    setTimeout(() => {
                        if (this.conversationActive && !this.isRecording) {
                            this.micStatus.textContent = 'Press and hold to talk';
                        }
                    }, 1000);
                }
                return;
            }
            const formData = new FormData();
            try {
                const wavBlob = await this.convertWebmToWav16kMono(audioBlob);
                if (wavBlob && wavBlob.size > 0) {
                    formData.append('audio_file', wavBlob, 'recording.wav');
                } else {
                    formData.append('audio_file', audioBlob, 'recording.webm');
                }
            } catch (e) {
                console.warn('WAV conversion failed, sending WEBM:', e);
                formData.append('audio_file', audioBlob, 'recording.webm');
            }
            // Start measuring round-trip (upload + server + download)
            this.lastTurnStartTime = Date.now();
            
            // Use low-latency fast path
            const response = await fetch(`/api/v1/conversation/${this.candidateId}/turn-fast`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to process voice turn');
            }
            
            const data = await response.json();
            
            // Show ASR debug (recognized text + confidence)
            if (data.asr && (data.asr.text || data.asr.text === '')) {
                console.log(`ASR: "${data.asr.text}" (conf ${Math.round((data.asr.confidence || 0)*100)}%)`);
                this.addToConversationLog('ASR', `${data.asr.text || ''} (${Math.round((data.asr.confidence || 0)*100)}%)`, false);
            }
            
            // Update metrics
            this.updateMetrics(data.metrics);
            
            // Update candidate profile early
            if (data.metrics && data.metrics.candidate_profile) {
                this.updateCandidateProfile(data.metrics.candidate_profile);
            }
            
            // Roles preview disabled
            
            // Play audio via background fetch (turn-fast returns no audio inline)
            const turnId = data.turn_id;
            
            // Update conversation log
            this.addToConversationLog('User', '[Voice input]', false);
            if (data.text) {
                this.addToConversationLog('Bot', data.text, true);
            } else {
                this.addToConversationLog('Bot', 'Audio response pending...', true);
            }
            
            // Debug logging
            console.log('API Response data:', data);
            console.log('Metrics:', data.metrics);
            
            // Check if conversation is complete
            if (data.conversation_complete) {
                this.conversationActive = false;
                this.micStatus.textContent = 'Interview completed!';
                this.updateUIForInactiveConversation();
                if (data.matches && Array.isArray(data.matches) && data.matches.length > 0) {
                    this.displayJobMatches(data.matches);
                } else {
                    // Fallback: fetch matches via API
                    try {
                        const mresp = await fetch(`/api/v1/conversation/${this.candidateId}/matches`);
                        if (mresp.ok) {
                            const mdata = await mresp.json();
                            if (mdata && Array.isArray(mdata.matches)) {
                                this.displayJobMatches(mdata.matches);
                            }
                        } else {
                            console.warn('Matches API returned non-OK status');
                        }
                    } catch (e) {
                        console.warn('Failed to fetch matches:', e);
                    }
                }
                // After matches are visible, play the final TTS
                if (turnId) {
                    await this.pollAndPlayTurnAudio(turnId);
                }
            } else {
                this.micStatus.textContent = 'Press and hold to talk';
                // For intermediate turns, play audio after UI updates
                if (turnId) {
                    await this.pollAndPlayTurnAudio(turnId);
                }
            }
            
        } catch (error) {
            console.error('Error processing recording:', error);
            this.showError('Failed to process recording: ' + error.message);
            this.micStatus.textContent = 'Error - press and hold to try again';
        }
    }

    async playAudioFromBase64(base64Audio) {
        return new Promise((resolve, reject) => {
            try {
                const audioBlob = this.base64ToBlob(base64Audio, 'audio/wav');
                const audioUrl = URL.createObjectURL(audioBlob);
                
                this.responseAudio.src = audioUrl;
                this.responseAudio.onended = () => {
                    URL.revokeObjectURL(audioUrl);
                    resolve();
                };
                this.responseAudio.onerror = () => {
                    URL.revokeObjectURL(audioUrl);
                    reject(new Error('Failed to play audio'));
                };
                
                this.responseAudio.play();
                
            } catch (error) {
                reject(error);
            }
        });
    }

    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }

    updateMetrics(metrics) {
        if (!metrics) return;
        
        // Response time
        if (this.lastTurnStartTime) {
            const roundTripMs = Date.now() - this.lastTurnStartTime;
            const serverMs = metrics.last_server_latency_ms || roundTripMs;
            const display = `${roundTripMs} ms` + (serverMs && serverMs !== roundTripMs ? ` (server ${serverMs} ms)` : '');
            this.responseTimeEl.textContent = display;
            
            // Color code based on target (round-trip)
            if (roundTripMs < 2000) {
                this.responseTimeEl.className = 'text-2xl font-bold text-green-600';
            } else if (roundTripMs < 3000) {
                this.responseTimeEl.className = 'text-2xl font-bold text-yellow-600';
            } else {
                this.responseTimeEl.className = 'text-2xl font-bold text-red-600';
            }
        }
        
        // Voice confidence
        if (metrics.avg_confidence !== undefined) {
            const confidence = Math.round(metrics.avg_confidence * 100);
            this.voiceConfidenceEl.textContent = `${confidence}%`;
            
            if (confidence >= 70) {
                this.voiceConfidenceEl.className = 'text-2xl font-bold text-green-600';
            } else if (confidence >= 50) {
                this.voiceConfidenceEl.className = 'text-2xl font-bold text-yellow-600';
            } else {
                this.voiceConfidenceEl.className = 'text-2xl font-bold text-red-600';
            }
        }
        
        // Progress
        if (metrics.completion_rate !== undefined) {
            const progress = Math.round(metrics.completion_rate * 100);
            this.progressBar.style.width = `${progress}%`;
            this.progressText.textContent = `Interview Progress: ${progress}%`;
            
            // Update fields collected
            const fieldsCompleted = Math.round(metrics.completion_rate * 7);
            this.fieldsCollectedEl.textContent = `${fieldsCompleted}/7`;
        }
    }

    renderLanguageTags(langs = [], otherLangs = []) {
        const container = this.profileElements.languages;
        if (!container) return;
        container.innerHTML = '';
        const cap = (s) => (typeof s === 'string' && s.length > 0) ? (s[0].toUpperCase() + s.slice(1)) : s;
        const makeTag = (label, cls) => {
            const span = document.createElement('span');
            span.className = `inline-block ${cls} text-xs px-2 py-1 rounded-full mr-2 mb-2`;
            span.textContent = label;
            return span;
        };
        if ((!langs || langs.length === 0) && (!otherLangs || otherLangs.length === 0)) {
            container.textContent = '--';
            return;
        }
        (langs || []).forEach(l => container.appendChild(makeTag(cap(l), 'bg-blue-100 text-blue-800')));
        (otherLangs || []).forEach(l => container.appendChild(makeTag(`Other: ${cap(l)}`, 'bg-gray-100 text-gray-800')));
    }

    updateCandidateProfile(profile) {
        console.log('Updating candidate profile:', profile);
        
        if (profile.pincode) {
            this.profileElements.pincode.textContent = profile.pincode;
        } else if (profile.locality) {
            this.profileElements.pincode.textContent = profile.locality;
        }
        if (profile.expected_salary) {
            this.profileElements.salary.textContent = `₹${profile.expected_salary.toLocaleString()}/month`;
        }
        if (profile.preferred_shift) {
            this.profileElements.shift.textContent = profile.preferred_shift.replace('_', ' ').toUpperCase();
        }
        this.renderLanguageTags(profile.languages || [], profile.other_languages || []);
        if (profile.has_two_wheeler !== undefined) {
            this.profileElements.twoWheeler.textContent = profile.has_two_wheeler ? 'Yes' : 'No';
        }
        if (profile.total_experience_months !== undefined) {
            console.log('Experience value:', profile.total_experience_months);
            const totalMonths = parseInt(profile.total_experience_months) || 0;
            const years = Math.floor(totalMonths / 12);
            const months = totalMonths % 12;
            console.log('Parsed experience - Total months:', totalMonths, 'Years:', years, 'Months:', months);
            
            if (years > 0) {
                if (months > 0) {
                    this.profileElements.experience.textContent = `${years} year${years > 1 ? 's' : ''} ${months} month${months !== 1 ? 's' : ''}`;
                } else {
                    this.profileElements.experience.textContent = `${years} year${years > 1 ? 's' : ''}`;
                }
            } else if (months > 0) {
                this.profileElements.experience.textContent = `${months} month${months !== 1 ? 's' : ''}`;
            } else {
                // Handle 0 months case (fresher)
                this.profileElements.experience.textContent = '0 months (Fresher)';
            }
        }
    }

    // Roles preview removed: no-op methods for compatibility
    async loadJobMatches() { return; }
    displayJobMatches(matches) {
        if (!this.jobMatches || !this.jobMatchesList) return;
        this.jobMatchesList.innerHTML = '';
        const list = Array.isArray(matches) ? matches : [];
        if (list.length === 0) {
            const empty = document.createElement('div');
            empty.className = 'text-gray-600 bg-gray-50 border border-dashed border-gray-300 rounded-lg p-6 text-center';
            empty.textContent = 'No matching jobs found right now. We will notify you when a suitable role is available.';
            this.jobMatchesList.appendChild(empty);
        } else {
            list.forEach((match, index) => {
                const matchElement = this.createJobMatchElement(match, index + 1);
                this.jobMatchesList.appendChild(matchElement);
            });
        }
        this.jobMatches.classList.remove('hidden');
    }

    createJobMatchElement(match, rank) {
        const matchScore = Math.round(match.match_score * 100);
        const job = match.job;
        
        const element = document.createElement('div');
        element.className = 'bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500';
        
        element.innerHTML = `
            <div class="flex justify-between items-start mb-4">
                <div>
                    <span class="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded-full mb-2">#${rank} Match</span>
                    <h4 class="text-xl font-bold text-gray-900">${job.title}</h4>
                    <p class="text-gray-600">${job.company} • ${job.locality}</p>
                </div>
                <div class="text-right">
                    <div class="text-2xl font-bold ${matchScore >= 80 ? 'text-green-600' : matchScore >= 60 ? 'text-yellow-600' : 'text-red-600'}">${matchScore}%</div>
                    <div class="text-sm text-gray-500">Match Score</div>
                </div>
            </div>
            
            <div class="mb-4">
                <p class="text-gray-700 font-medium">${match.rationale}</p>
            </div>
            
            <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <span class="text-sm text-gray-500">Salary Range</span>
                    <p class="font-semibold">₹${job.salary_min.toLocaleString()} - ₹${job.salary_max.toLocaleString()}</p>
                </div>
                <div>
                    <span class="text-sm text-gray-500">Contact</span>
                    <p class="font-semibold">${job.contact_number}</p>
                </div>
            </div>
            
            ${match.strengths.length > 0 ? `
                <div class="mb-3">
                    <h5 class="font-semibold text-green-700 mb-2">✓ Strengths:</h5>
                    <ul class="text-sm text-green-600 space-y-1">
                        ${match.strengths.map(strength => `<li>• ${strength}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            ${match.concerns.length > 0 ? `
                <div class="mb-4">
                    <h5 class="font-semibold text-orange-700 mb-2">⚠ Considerations:</h5>
                    <ul class="text-sm text-orange-600 space-y-1">
                        ${match.concerns.map(concern => `<li>• ${concern}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
            
            <div class="flex justify-between items-center pt-4 border-t">
                <div class="text-sm text-gray-500">
                    ${job.description}
                </div>
                <button class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors">
                    Apply Now
                </button>
            </div>
        `;
        
        return element;
    }

    updateRecordingUI(recording) {
        if (recording) {
            this.micButton.classList.add('recording', 'bg-red-500');
            this.micButton.classList.remove('bg-blue-500');
            this.micStatus.textContent = 'Listening... release to send';
            this.audioVisualizer.classList.remove('hidden');
        } else {
            this.micButton.classList.remove('recording', 'bg-red-500');
            this.micButton.classList.add('bg-blue-500');
            this.audioVisualizer.classList.add('hidden');
            if (this.conversationActive) {
                this.micStatus.textContent = 'Press and hold to talk';
            }
        }
    }

    updateUIForActiveConversation() {
        this.startBtn.disabled = true;
        this.pauseBtn.disabled = false;
        this.stopBtn.disabled = false;
        this.micButton.classList.remove('cursor-not-allowed');
    }

    updateUIForInactiveConversation() {
        this.startBtn.disabled = false;
        this.pauseBtn.disabled = true;
        this.stopBtn.disabled = true;
        this.pauseBtn.innerHTML = '<i class="fas fa-pause mr-2"></i>Pause';
    }

    updateConnectionStatus(status, colorClass) {
        this.connectionStatus.textContent = status;
        this.connectionStatus.className = `font-semibold ${colorClass}`;
    }

    addToConversationLog(speaker, message, isAudio = false) {
        const logEntry = document.createElement('div');
        logEntry.className = `p-3 rounded-lg ${speaker === 'Bot' ? 'bg-blue-50 border-l-4 border-blue-500' : 'bg-gray-50 border-l-4 border-gray-500'}`;
        
        const timestamp = new Date().toLocaleTimeString();
        logEntry.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <span class="font-semibold text-sm ${speaker === 'Bot' ? 'text-blue-700' : 'text-gray-700'}">${speaker}</span>
                    ${isAudio ? '<i class="fas fa-volume-up ml-2 text-xs"></i>' : ''}
                </div>
                <span class="text-xs text-gray-500">${timestamp}</span>
            </div>
            <p class="mt-1 text-sm">${message}</p>
        `;
        
        // Clear placeholder text if it exists
        if (this.conversationLog.querySelector('.text-gray-500.text-center')) {
            this.conversationLog.innerHTML = '';
        }
        
        this.conversationLog.appendChild(logEntry);
        this.conversationLog.scrollTop = this.conversationLog.scrollHeight;
    }

    async runJobMatchingDemo() {
        try {
            this.addToConversationLog('System', 'Running job matching demo...', false);
            
            const response = await fetch('/api/v1/demo/job-match', {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to run demo');
            }
            
            const data = await response.json();
            
            // Show candidate profile
            this.candidateProfile.classList.remove('hidden');
            this.profileElements.pincode.textContent = data.candidate.pincode;
            this.profileElements.salary.textContent = `₹${data.candidate.expected_salary.toLocaleString()}/month`;
            this.profileElements.shift.textContent = data.candidate.preferred_shift.toUpperCase();
            this.renderLanguageTags(data.candidate.languages || [], data.candidate.other_languages || []);
            this.profileElements.twoWheeler.textContent = data.candidate.has_two_wheeler ? 'Yes' : 'No';
            this.profileElements.experience.textContent = `${data.candidate.experience_months} months`;
            
            // Roles preview disabled
            
        } catch (error) {
            console.error('Demo error:', error);
            this.showError('Failed to run job matching demo: ' + error.message);
        }
    }

    showError(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'fixed top-4 right-4 bg-red-500 text-white p-4 rounded-lg shadow-lg z-50 max-w-md';
        errorDiv.innerHTML = `
            <div class="flex justify-between items-start">
                <div>
                    <h4 class="font-bold">Error</h4>
                    <p class="text-sm mt-1">${message}</p>
                </div>
                <button class="ml-4 text-red-200 hover:text-white" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        document.body.appendChild(errorDiv);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 10000);
    }

    renderRolesPreview() {
        if (!this.rolesPreviewList) return;
        this.rolesPreviewList.innerHTML = '';
        this.availableRoles.forEach(role => {
            const li = document.createElement('li');
            li.textContent = role;
            this.rolesPreviewList.appendChild(li);
        });
    }

    async pollAndPlayTurnAudio(turnId) {
        // Poll for background audio; play when ready
        const maxAttempts = 20; // ~10s at 500ms interval
        const delay = (ms) => new Promise(res => setTimeout(res, ms));
        for (let i = 0; i < maxAttempts; i++) {
            try {
                const resp = await fetch(`/api/v1/conversation/${this.candidateId}/turn-audio/${turnId}`);
                if (resp.ok) {
                    const payload = await resp.json();
                    if (payload.ready && payload.audio_data) {
                        await this.playAudioFromBase64(payload.audio_data);
                        return;
                    }
                }
            } catch (e) {
                console.warn('Audio fetch attempt failed:', e);
            }
            await delay(500);
        }
        console.warn('Audio not ready in time; skipping playback for this turn.');
    }

    async convertWebmToWav16kMono(webmBlob) {
        const arrayBuffer = await webmBlob.arrayBuffer();
        const ac = new (window.AudioContext || window.webkitAudioContext)();
        try {
            const audioBuffer = await ac.decodeAudioData(arrayBuffer.slice(0));
            const duration = audioBuffer.duration;
            const offline = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, Math.ceil(16000 * duration), 16000);
            // Mix to mono
            const monoBuffer = offline.createBuffer(1, audioBuffer.length, audioBuffer.sampleRate);
            const channelData = monoBuffer.getChannelData(0);
            const numChannels = audioBuffer.numberOfChannels;
            for (let ch = 0; ch < numChannels; ch++) {
                const data = audioBuffer.getChannelData(ch);
                for (let i = 0; i < data.length; i++) {
                    channelData[i] += data[i] / numChannels;
                }
            }
            const src = offline.createBufferSource();
            src.buffer = monoBuffer;
            src.connect(offline.destination);
            src.start(0);
            const rendered = await offline.startRendering();
            const wavBuffer = this.audioBufferToWav(rendered);
            return new Blob([wavBuffer], { type: 'audio/wav' });
        } finally {
            try { ac.close(); } catch (_) {}
        }
    }

    audioBufferToWav(buffer) {
        const numOfChannels = buffer.numberOfChannels;
        const sampleRate = buffer.sampleRate;
        const format = 1; // PCM
        const bitDepth = 16;
        const numFrames = buffer.length;
        const bytesPerSample = bitDepth / 8;
        const blockAlign = numOfChannels * bytesPerSample;
        const byteRate = sampleRate * blockAlign;
        const dataLength = numFrames * blockAlign;
        const bufferLength = 44 + dataLength;
        const arrayBuffer = new ArrayBuffer(bufferLength);
        const view = new DataView(arrayBuffer);

        let offset = 0;
        const writeString = (s) => { for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i)); offset += s.length; };
        const writeUint32 = (d) => { view.setUint32(offset, d, true); offset += 4; };
        const writeUint16 = (d) => { view.setUint16(offset, d, true); offset += 2; };

        writeString('RIFF');
        writeUint32(36 + dataLength);
        writeString('WAVE');
        writeString('fmt ');
        writeUint32(16);
        writeUint16(format);
        writeUint16(numOfChannels);
        writeUint32(sampleRate);
        writeUint32(byteRate);
        writeUint16(blockAlign);
        writeUint16(bitDepth);
        writeString('data');
        writeUint32(dataLength);

        const channelData = [];
        for (let ch = 0; ch < numOfChannels; ch++) {
            channelData.push(buffer.getChannelData(ch));
        }
        let idx = 0;
        for (let i = 0; i < numFrames; i++) {
            for (let ch = 0; ch < numOfChannels; ch++) {
                let sample = Math.max(-1, Math.min(1, channelData[ch][i]));
                view.setInt16(44 + idx, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
                idx += 2;
            }
        }
        return arrayBuffer;
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VoiceBot();
}); 