class VoiceBot {
    constructor() {
        this.candidateId = null;
        this.isRecording = false;
        this.isPaused = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.conversationActive = false;
        this.lastTurnStartTime = null;
        
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
        
        // Job matches
        this.jobMatches = document.getElementById('jobMatches');
        this.jobMatchesList = document.getElementById('jobMatchesList');
    }

    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startConversation());
        this.demoBtn.addEventListener('click', () => this.runJobMatchingDemo());
        this.pauseBtn.addEventListener('click', () => this.pauseConversation());
        this.stopBtn.addEventListener('click', () => this.stopConversation());
        this.micButton.addEventListener('click', () => this.toggleRecording());
        
        // Audio playback ended - start listening again
        this.responseAudio.addEventListener('ended', () => {
            if (this.conversationActive && !this.isPaused) {
                this.startListening();
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
            
            // Start listening after greeting
            setTimeout(() => this.startListening(), 1000);
            
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
            this.startListening();
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
        
        // Show final results if conversation was completed
        if (this.candidateId) {
            this.loadJobMatches();
        }
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
            this.lastTurnStartTime = Date.now();
            this.mediaRecorder.start();
            
            this.updateRecordingUI(true);
            
            // Auto-stop recording after 10 seconds
            setTimeout(() => {
                if (this.isRecording) {
                    this.stopRecording();
                }
            }, 10000);
            
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

    async processRecording() {
        if (this.audioChunks.length === 0) return;
        
        try {
            this.micStatus.textContent = 'Processing your voice...';
            
            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
            const formData = new FormData();
            formData.append('audio_file', audioBlob, 'recording.webm');
            
            const response = await fetch(`/api/v1/conversation/${this.candidateId}/turn`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to process voice turn');
            }
            
            const data = await response.json();
            
            // Update metrics
            this.updateMetrics(data.metrics);
            
            // Play response audio
            await this.playAudioFromBase64(data.audio_data);
            
            // Update conversation log
            this.addToConversationLog('User', '[Voice input]', false);
            if (data.text) {
                this.addToConversationLog('Bot', data.text, true);
            } else {
                this.addToConversationLog('Bot', 'Audio response played', true);
            }
            
            // Update candidate profile
            if (data.metrics && data.metrics.candidate_profile) {
                this.updateCandidateProfile(data.metrics.candidate_profile);
            }
            
            // Check if conversation is complete
            if (data.conversation_complete) {
                this.conversationActive = false;
                this.micStatus.textContent = 'Interview completed!';
                this.updateUIForInactiveConversation();
                await this.loadJobMatches();
            } else {
                this.micStatus.textContent = 'Listening...';
            }
            
        } catch (error) {
            console.error('Error processing recording:', error);
            this.showError('Failed to process recording: ' + error.message);
            this.micStatus.textContent = 'Error - click to try again';
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
            const responseTime = Date.now() - this.lastTurnStartTime;
            this.responseTimeEl.textContent = `${responseTime} ms`;
            
            // Color code based on target
            if (responseTime < 2000) {
                this.responseTimeEl.className = 'text-2xl font-bold text-green-600';
            } else if (responseTime < 3000) {
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

    updateCandidateProfile(profile) {
        if (profile.pincode) {
            this.profileElements.pincode.textContent = profile.pincode;
        }
        if (profile.expected_salary) {
            this.profileElements.salary.textContent = `₹${profile.expected_salary.toLocaleString()}/month`;
        }
        if (profile.preferred_shift) {
            this.profileElements.shift.textContent = profile.preferred_shift.replace('_', ' ').toUpperCase();
        }
        if (profile.languages && profile.languages.length > 0) {
            this.profileElements.languages.textContent = profile.languages.join(', ').toUpperCase();
        }
        if (profile.has_two_wheeler !== undefined) {
            this.profileElements.twoWheeler.textContent = profile.has_two_wheeler ? 'Yes' : 'No';
        }
        if (profile.total_experience_months !== undefined) {
            const years = Math.floor(profile.total_experience_months / 12);
            const months = profile.total_experience_months % 12;
            if (years > 0) {
                this.profileElements.experience.textContent = `${years} year${years > 1 ? 's' : ''} ${months} month${months > 1 ? 's' : ''}`;
            } else {
                this.profileElements.experience.textContent = `${months} month${months > 1 ? 's' : ''}`;
            }
        }
    }

    async loadJobMatches() {
        try {
            const response = await fetch(`/api/v1/conversation/${this.candidateId}/matches`);
            
            if (!response.ok) {
                throw new Error('Failed to load job matches');
            }
            
            const data = await response.json();
            this.displayJobMatches(data.matches);
            
        } catch (error) {
            console.error('Error loading job matches:', error);
            this.showError('Failed to load job matches: ' + error.message);
        }
    }

    displayJobMatches(matches) {
        this.jobMatchesList.innerHTML = '';
        
        matches.forEach((match, index) => {
            const matchElement = this.createJobMatchElement(match, index + 1);
            this.jobMatchesList.appendChild(matchElement);
        });
        
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
            this.micStatus.textContent = 'Recording... (speak now)';
            this.audioVisualizer.classList.remove('hidden');
        } else {
            this.micButton.classList.remove('recording', 'bg-red-500');
            this.micButton.classList.add('bg-blue-500');
            this.audioVisualizer.classList.add('hidden');
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
            this.profileElements.languages.textContent = data.candidate.languages.join(', ').toUpperCase();
            this.profileElements.twoWheeler.textContent = data.candidate.has_two_wheeler ? 'Yes' : 'No';
            this.profileElements.experience.textContent = `${data.candidate.experience_months} months`;
            
            // Show job matches
            this.displayJobMatches(data.matches);
            
            this.addToConversationLog('System', `Found ${data.matches.length} job matches from ${data.total_jobs_considered} total jobs`, false);
            
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
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VoiceBot();
}); 