package com.extrahand.cardtrainer

import android.content.Context
import com.extrahand.cards.sync.DatasetSource
import com.extrahand.cards.sync.GitHubConfig

/** Persisted app settings. The token lives in app-private storage (single-user device). */
data class TrainerSettings(
    val owner: String = DatasetSource.DEFAULT.owner,
    val repo: String = DatasetSource.DEFAULT.repo,
    val branch: String = DatasetSource.DEFAULT.branch,
    val path: String = DatasetSource.DEFAULT.path,
    val token: String = "",
    /** Push automatically once this many changes are pending (0 = manual only). */
    val autoSyncEvery: Int = 10,
) {
    val canSync get() = token.isNotBlank() && owner.isNotBlank() && repo.isNotBlank()
    fun gitHub() = GitHubConfig(owner.trim(), repo.trim(), branch.trim().ifEmpty { "main" }, path.trim('/', ' '), token.trim())

    companion object {
        private const val PREFS = "trainer_settings"

        fun load(ctx: Context): TrainerSettings {
            val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val d = TrainerSettings()
            return TrainerSettings(
                owner = p.getString("owner", d.owner)!!,
                repo = p.getString("repo", d.repo)!!,
                branch = p.getString("branch", d.branch)!!,
                path = p.getString("path", d.path)!!,
                token = p.getString("token", d.token)!!,
                autoSyncEvery = p.getInt("autoSyncEvery", d.autoSyncEvery),
            )
        }

        fun save(ctx: Context, s: TrainerSettings) {
            ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
                .putString("owner", s.owner).putString("repo", s.repo).putString("branch", s.branch)
                .putString("path", s.path).putString("token", s.token).putInt("autoSyncEvery", s.autoSyncEvery)
                .apply()
        }
    }
}
