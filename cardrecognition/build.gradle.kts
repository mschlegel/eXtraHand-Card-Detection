plugins {
    id("com.android.library")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.serialization")
    `maven-publish`
}

android {
    namespace = "com.extrahand.cards"
    compileSdk = 35
    defaultConfig {
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    publishing { singleVariant("release") { withSourcesJar() } }
}

// JitPack: implementation("com.github.mschlegel:eXtraHand-Card-Detection:<tag or commit>")
afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])
                groupId = "com.github.mschlegel"
                artifactId = "cardrecognition"
                version = (findProperty("libVersion") as String?) ?: "0.1.0"
            }
        }
    }
}

dependencies {
    api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    api("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    // CameraX is compileOnly: only needed if the host app uses CardFrameAnalyzer.
    compileOnly("androidx.camera:camera-core:1.4.1")
    testImplementation("junit:junit:4.13.2")
}
